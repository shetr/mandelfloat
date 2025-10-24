//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    asset::RenderAssetUsages, input::mouse::MouseWheel, prelude::*, render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin}, render_asset::RenderAssets, render_graph::{self, RenderGraph, RenderLabel}, render_resource::{
            binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, texture_storage_2d, uniform_buffer},
            *,
        }, renderer::{RenderContext, RenderDevice, RenderQueue}, texture::GpuImage, Render, RenderApp, RenderSystems
    }, ui::RelativeCursorPosition
    
};
use bevy_egui::{egui, input::egui_wants_any_input, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use std::{borrow::Cow, num::NonZero};

/// This example uses a shader source file from the assets subdirectory
const ITERATIONS_SHADER_ASSET_PATH: &str = "shaders/iterations.wgsl";
const VISUALIZE_SHADER_ASSET_PATH: &str = "shaders/visualize.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: UVec2 = UVec2::new(1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;
const SCROLL_ZOOM_SPEED: f32 = 0.02;
const KEY_ZOOM_SPEED: f32 = 0.25;
const MOVE_SPEED: f32 = 1.0;
const ROTATION_SPEED: f32 = 0.5;
const SPEED_MULTIPILER: f32 = 2.0;

//const BUFFER_DATA: [IterationResult; (SIZE.x * SIZE.y) as usize] = [IterationResult { i:0, z: Vec2::ZERO }; (SIZE.x * SIZE.y) as usize];

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            MandelfloatComputePlugin,
            EguiPlugin::default(),
        ))
        .add_systems(Startup, setup)
        .add_systems(EguiPrimaryContextPass, ui_update)
        .add_systems(Update, update_input.run_if(not(egui_wants_any_input)))
        .run();
}

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image_handle = images.add(image);

    commands.spawn(Node {
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        align_items: AlignItems::Center,
        justify_content: JustifyContent::Center,
        flex_direction: FlexDirection::Column,
        ..default()
    }).with_children(|parent| {
        parent.spawn((
            Node{
                width: Val::Px(SIZE.x as f32),
                height: Val::Px(SIZE.y as f32),
                ..default()
            },
            ImageNode {
                image: image_handle.clone(),
                ..default()
            },
            Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
            RelativeCursorPosition::default(),
        ));
    });

    commands.spawn(Camera2d);

    commands.insert_resource(OutputImage {
        texture: image_handle,
    });

    commands.insert_resource(IterationsUniforms {
        transform: Mat3::IDENTITY,
        dimensions: SIZE,
    });
    
    commands.insert_resource(VisualizeUniforms {
        test_color: LinearRgba::WHITE,
    });

    commands.insert_resource(MandelfloatData {
        last_cursor_position: None,
        position: vec2(0.0, 0.0),
        zoom: 0.0,
        rotation_angle: 0.0,
    });
}

#[derive(Resource)]
struct MandelfloatData
{
    last_cursor_position: Option<Vec2>,
    position: Vec2,
    zoom: f32,
    rotation_angle: f32,
}

fn ui_update(mut contexts: EguiContexts) -> Result {
    egui::Window::new("Hello").show(contexts.ctx_mut()?, |ui| {
        ui.label("world");
    });
    Ok(())
}

fn compute_scale(zoom: f32) -> f32
{
    10.0f32.powf(-zoom)
}

fn compute_transform(position: Vec2, scale: f32, rotation_angle: f32) -> Mat3
{
    Mat3::from_scale_angle_translation(vec2(scale, scale), rotation_angle, position)
}

fn update_input(
    mut iterations_uniforms: ResMut<IterationsUniforms>,
    mut data: ResMut<MandelfloatData>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    relative_cursor_position: Single<&RelativeCursorPosition>,
    mut mouse_wheel_reader: MessageReader<MouseWheel>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    for mouse_wheel in mouse_wheel_reader.read() {
        let zoom_increment = mouse_wheel.y.clamp(-3.0, 3.0);
        data.zoom += zoom_increment * SCROLL_ZOOM_SPEED;
    }

    let speed_mul = if keyboard_input.pressed(KeyCode::ShiftLeft) { SPEED_MULTIPILER } else { 1.0 };

    if keyboard_input.pressed(KeyCode::NumpadAdd) || keyboard_input.pressed(KeyCode::KeyM) {
        data.zoom += KEY_ZOOM_SPEED * speed_mul * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::NumpadSubtract) || keyboard_input.pressed(KeyCode::KeyN) {
        data.zoom -= KEY_ZOOM_SPEED * speed_mul * time.delta_secs();
    }
    
    if keyboard_input.pressed(KeyCode::KeyQ) {
        data.rotation_angle += ROTATION_SPEED * speed_mul * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::KeyE) {
        data.rotation_angle -= ROTATION_SPEED * speed_mul * time.delta_secs();
    }
    
    let scale = compute_scale(data.zoom);
    let rot_scale_mat = Mat2::from_scale_angle(vec2(scale, scale), data.rotation_angle);

    if mouse_button.just_pressed(MouseButton::Left) || mouse_button.just_released(MouseButton::Left) {
        data.last_cursor_position = None;
    } else if mouse_button.pressed(MouseButton::Left) {
        if let Some(relative_cursor_position) = relative_cursor_position.normalized {
            let ratio = vec2((SIZE.x as f32) / (SIZE.y as f32), -1.0);
            let curr_cursor_position = relative_cursor_position * 2.0 * ratio;
            if let Some(last_cursor_position) = data.last_cursor_position {
                data.position += rot_scale_mat * (last_cursor_position - curr_cursor_position);
            }
            data.last_cursor_position = Some(curr_cursor_position);
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowLeft) || keyboard_input.pressed(KeyCode::KeyA) {
        data.position -= rot_scale_mat.col(0) * MOVE_SPEED * speed_mul * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowRight) || keyboard_input.pressed(KeyCode::KeyD) {
        data.position += rot_scale_mat.col(0) * MOVE_SPEED * speed_mul * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowDown) || keyboard_input.pressed(KeyCode::KeyS) {
        data.position -= rot_scale_mat.col(1) * MOVE_SPEED * speed_mul * time.delta_secs();
    }
    if keyboard_input.pressed(KeyCode::ArrowUp) || keyboard_input.pressed(KeyCode::KeyW) {
        data.position += rot_scale_mat.col(1) * MOVE_SPEED * speed_mul * time.delta_secs();
    }

    iterations_uniforms.transform = compute_transform(data.position, scale, data.rotation_angle);
}

struct MandelfloatComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MandelfloatLabel;

impl Plugin for MandelfloatComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(
            (ExtractResourcePlugin::<OutputImage>::default(),
            ExtractResourcePlugin::<IterationsUniforms>::default(),
            ExtractResourcePlugin::<VisualizeUniforms>::default(),
        ));
        let render_app = app.sub_app_mut(RenderApp);
        render_app.add_systems(
            Render,
            prepare_bind_group.in_set(RenderSystems::PrepareBindGroups),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(MandelfloatLabel, MandelfloatNode::default());
        render_graph.add_node_edge(MandelfloatLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<MandelfloatPipeline>();
    }
}

#[derive(Resource, Clone, ExtractResource)]
struct OutputImage {
    texture: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct IterationsUniforms {
    transform: Mat3,
    dimensions: UVec2,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct VisualizeUniforms {
    test_color: LinearRgba,
}

#[derive(Resource, Clone, Copy, ExtractResource, ShaderType)]
struct IterationResult {
    z: Vec2,
    i: u32,
}

#[derive(Resource)]
struct MandelfloatBindGroups
{
    iter: [BindGroup; 2],
    vis: [BindGroup; 2],
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<MandelfloatPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    output_image: Res<OutputImage>,
    iterations_uniforms: Res<IterationsUniforms>,
    visualize_uniforms: Res<VisualizeUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view = gpu_images.get(&output_image.texture).unwrap();
    
    let mut iter_uniform_buffer = UniformBuffer::from(iterations_uniforms.into_inner());
    iter_uniform_buffer.write_buffer(&render_device, &queue);
    
    let mut vis_uniform_buffer = UniformBuffer::from(visualize_uniforms.into_inner());
    vis_uniform_buffer.write_buffer(&render_device, &queue);

    let buffer_data = vec![IterationResult { i:0, z: Vec2::ZERO }; (SIZE.x * SIZE.y) as usize];
    let mut buffer1 = StorageBuffer::from(buffer_data.clone());
    let mut buffer2 = StorageBuffer::from(buffer_data);
    buffer1.write_buffer(&render_device, &queue);
    buffer2.write_buffer(&render_device, &queue);

    let iter_bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.iter_bind_group_layout,
        &BindGroupEntries::sequential((
            &buffer1,
            &buffer2,
            &iter_uniform_buffer,
        )),
    );
    let iter_bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.iter_bind_group_layout,
        &BindGroupEntries::sequential((
            &buffer2,
            &buffer1,
            &iter_uniform_buffer,
        )),
    );
    let vis_bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.vis_bind_group_layout,
        &BindGroupEntries::sequential((
            &buffer2,
            &view.texture_view,
            &vis_uniform_buffer,
        )),
    );
    let vis_bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.vis_bind_group_layout,
        &BindGroupEntries::sequential((
            &buffer1,
            &view.texture_view,
            &vis_uniform_buffer,
        )),
    );
    commands.insert_resource(MandelfloatBindGroups {
        iter: [iter_bind_group_0, iter_bind_group_1],
        vis: [vis_bind_group_0, vis_bind_group_1],
    });
}

#[derive(Resource)]
struct MandelfloatPipeline {
    iter_bind_group_layout: BindGroupLayout,
    vis_bind_group_layout: BindGroupLayout,
    iter_init_pipeline: CachedComputePipelineId,
    iter_update_pipeline: CachedComputePipelineId,
    vis_pipeline: CachedComputePipelineId,
}

impl FromWorld for MandelfloatPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let storage_size = NonZero::<u64>::new((SIZE.x * SIZE.y * (size_of::<IterationResult>() as u32)) as u64);
        let iter_bind_group_layout = render_device.create_bind_group_layout(
            "IterLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, storage_size),
                    storage_buffer_sized(false, storage_size),
                    uniform_buffer::<IterationsUniforms>(false),
                ),
            ),
        );
        let vis_bind_group_layout = render_device.create_bind_group_layout(
            "VisLayout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only_sized(false, storage_size),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<VisualizeUniforms>(false),
                ),
            ),
        );
        let iter_shader = world.load_asset(ITERATIONS_SHADER_ASSET_PATH);
        let vis_shader = world.load_asset(VISUALIZE_SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let iter_init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![iter_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: iter_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("init")),
            zero_initialize_workgroup_memory: false,
        });
        let iter_update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![iter_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: iter_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("update")),
            zero_initialize_workgroup_memory: false,
        });
        let vis_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![vis_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: vis_shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("visualize")),
            zero_initialize_workgroup_memory: false,
        });

        MandelfloatPipeline {
            iter_bind_group_layout,
            vis_bind_group_layout,
            iter_init_pipeline,
            iter_update_pipeline,
            vis_pipeline,
        }
    }
}

#[derive(PartialEq, Eq)]
enum MandelfloatState {
    Loading,
    Init,
    Update(usize),
}

struct MandelfloatNode {
    state: MandelfloatState,
}

impl Default for MandelfloatNode {
    fn default() -> Self {
        Self {
            state: MandelfloatState::Loading,
        }
    }
}

impl render_graph::Node for MandelfloatNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<MandelfloatPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            MandelfloatState::Loading => {
                let mut all_ok = true;
                match pipeline_cache.get_compute_pipeline_state(pipeline.iter_init_pipeline) {
                    CachedPipelineState::Ok(_) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{ITERATIONS_SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {
                        all_ok = false;
                    }
                }
                match pipeline_cache.get_compute_pipeline_state(pipeline.vis_pipeline) {
                    CachedPipelineState::Ok(_) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{VISUALIZE_SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {
                        all_ok = false;
                    }
                }
                if all_ok {
                    self.state = MandelfloatState::Init;
                }
            }
            MandelfloatState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.iter_update_pipeline)
                {
                    if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.vis_pipeline)
                    {
                        self.state = MandelfloatState::Update(1);
                    }
                }
            }
            MandelfloatState::Update(0) => {
                self.state = MandelfloatState::Update(1);
            }
            MandelfloatState::Update(1) => {
                self.state = MandelfloatState::Update(0);
            }
            MandelfloatState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = world.resource::<MandelfloatBindGroups>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<MandelfloatPipeline>();

        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            // select the pipeline based on the current state
            match self.state {
                MandelfloatState::Loading => {}
                MandelfloatState::Init => {
                    let init_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.iter_init_pipeline)
                        .unwrap();
                    pass.set_bind_group(0, &bind_groups.iter[0], &[]);
                    pass.set_pipeline(init_pipeline);
                    pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
                }
                MandelfloatState::Update(index) => {
                    let update_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.iter_update_pipeline)
                        .unwrap();
                    pass.set_bind_group(0, &bind_groups.iter[index], &[]);
                    pass.set_pipeline(update_pipeline);
                    pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
                }
            }
        }
        {
            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());
            
            match self.state {
                MandelfloatState::Loading => {}
                MandelfloatState::Init => {
                    let vis_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.vis_pipeline)
                        .unwrap();
                    pass.set_bind_group(0, &bind_groups.vis[0], &[]);
                    pass.set_pipeline(vis_pipeline);
                    pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
                }
                MandelfloatState::Update(index) => {
                    let vis_pipeline = pipeline_cache
                        .get_compute_pipeline(pipeline.vis_pipeline)
                        .unwrap();
                    pass.set_bind_group(0, &bind_groups.vis[index], &[]);
                    pass.set_pipeline(vis_pipeline);
                    pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
                }
            }
        }
        

        Ok(())
    }
}