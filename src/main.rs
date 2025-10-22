//! A compute shader that simulates Conway's Game of Life.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::{texture_storage_2d, uniform_buffer},
            *,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    }, ui::RelativeCursorPosition,
    
};
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};
use std::borrow::Cow;

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/iterations.wgsl";

const DISPLAY_FACTOR: u32 = 1;
const SIZE: UVec2 = UVec2::new(1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

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
        .add_systems(Update, (update_input, switch_textures).chain())
        .add_systems(EguiPrimaryContextPass, ui_update)
        .run();
}

fn setup(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let image0 = images.add(image.clone());
    let image1 = images.add(image);

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
                image: image0.clone(),
                ..default()
            },
            Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
            RelativeCursorPosition::default(),
        ));
    });

    commands.spawn(Camera2d);

    commands.insert_resource(MandelfloatImages {
        texture_a: image0,
        texture_b: image1,
    });

    commands.insert_resource(MandelfloatUniforms {
        test_color: LinearRgba::WHITE,
        positon: vec2(0.0, 0.0),
        scale: 1.0,
    });

    commands.insert_resource(MandelfloatData {
        last_cursor_position: None
    });
}

#[derive(Resource)]
struct MandelfloatData
{
    last_cursor_position: Option<Vec2>
}

fn ui_update(mut contexts: EguiContexts) -> Result {
    egui::Window::new("Hello").show(contexts.ctx_mut()?, |ui| {
        ui.label("world");
    });
    Ok(())
}

// Switch texture to display every frame to show the one that was written to most recently.
fn switch_textures(images: Res<MandelfloatImages>, mut img_node: Single<&mut ImageNode>) {
    if img_node.image == images.texture_a {
        img_node.image = images.texture_b.clone();
    } else {
        img_node.image = images.texture_a.clone();
    }
}

fn update_input(
    mut uniforms: ResMut<MandelfloatUniforms>,
    mut data: ResMut<MandelfloatData>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    relative_cursor_position: Single<&RelativeCursorPosition>,
) {
    if mouse_button.just_pressed(MouseButton::Left) || mouse_button.just_released(MouseButton::Left) {
        data.last_cursor_position = None;
    } else if mouse_button.pressed(MouseButton::Left) {
        if let Some(relative_cursor_position) = relative_cursor_position.normalized {
            let ratio = vec2((SIZE.x as f32) / (SIZE.y as f32), 1.0);
            let curr_cursor_position = relative_cursor_position * 2.0 * ratio;
            if let Some(last_cursor_position) = data.last_cursor_position {
                uniforms.positon += last_cursor_position - curr_cursor_position;
            }
            data.last_cursor_position = Some(curr_cursor_position);
        }
    }

}

struct MandelfloatComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct MandelfloatLabel;

impl Plugin for MandelfloatComputePlugin {
    fn build(&self, app: &mut App) {
        // Extract the game of life image resource from the main world into the render world
        // for operation on by the compute shader and display on the sprite.
        app.add_plugins(
            (ExtractResourcePlugin::<MandelfloatImages>::default(),
            ExtractResourcePlugin::<MandelfloatUniforms>::default(),
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
struct MandelfloatImages {
    texture_a: Handle<Image>,
    texture_b: Handle<Image>,
}

#[derive(Resource, Clone, ExtractResource, ShaderType)]
struct MandelfloatUniforms {
    test_color: LinearRgba,
    positon: Vec2,
    scale: f32,
}

#[derive(Resource)]
struct MandelfloatImageBindGroups([BindGroup; 2]);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<MandelfloatPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mandelfloat_images: Res<MandelfloatImages>,
    mandelfloat_uniforms: Res<MandelfloatUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_a = gpu_images.get(&mandelfloat_images.texture_a).unwrap();
    let view_b = gpu_images.get(&mandelfloat_images.texture_b).unwrap();
    
    let mut uniform_buffer = UniformBuffer::from(mandelfloat_uniforms.into_inner());
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(MandelfloatImageBindGroups([bind_group_0, bind_group_1]));
}

#[derive(Resource)]
struct MandelfloatPipeline {
    texture_bind_group_layout: BindGroupLayout,
    init_pipeline: CachedComputePipelineId,
    update_pipeline: CachedComputePipelineId,
}

impl FromWorld for MandelfloatPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let texture_bind_group_layout = render_device.create_bind_group_layout(
            "MandelfloatImages",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                    texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                    uniform_buffer::<MandelfloatUniforms>(false),
                ),
            ),
        );
        let shader = world.load_asset(SHADER_ASSET_PATH);
        let pipeline_cache = world.resource::<PipelineCache>();
        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: vec![],
            entry_point: Some(Cow::from("init")),
            zero_initialize_workgroup_memory: false,
        });
        let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: None,
            layout: vec![texture_bind_group_layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs: vec![],
            entry_point: Some(Cow::from("update")),
            zero_initialize_workgroup_memory: false,
        });

        MandelfloatPipeline {
            texture_bind_group_layout,
            init_pipeline,
            update_pipeline,
        }
    }
}

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
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = MandelfloatState::Init;
                    }
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing assets/{SHADER_ASSET_PATH}:\n{err}")
                    }
                    _ => {}
                }
            }
            MandelfloatState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = MandelfloatState::Update(1);
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
        let bind_groups = &world.resource::<MandelfloatImageBindGroups>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<MandelfloatPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            MandelfloatState::Loading => {}
            MandelfloatState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
            MandelfloatState::Update(index) => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}