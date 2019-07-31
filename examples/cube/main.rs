use log::info;
use std::mem;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

#[allow(dead_code)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

pub fn load_glsl(code: &str, stage: ShaderStage) -> Vec<u8> {
    use std::io::Read;

    let ty = match stage {
        ShaderStage::Vertex => glsl_to_spirv::ShaderType::Vertex,
        ShaderStage::Fragment => glsl_to_spirv::ShaderType::Fragment,
        ShaderStage::Compute => glsl_to_spirv::ShaderType::Compute,
    };

    let mut output = glsl_to_spirv::compile(&code, ty).unwrap();
    let mut spv = Vec::new();
    output.read_to_end(&mut spv).unwrap();
    spv
}

#[derive(Clone, Copy)]
struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

#[derive(Clone, Copy)]
struct Instance {
    _pos: [f32; 4],
    _size: [f32; 4],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
    let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 10.0);
    let mx_view = cgmath::Matrix4::look_at(
        cgmath::Point3::new(1.5f32, -5.0, 3.0),
        cgmath::Point3::new(0f32, 0.0, 0.0),
        cgmath::Vector3::unit_z(),
    );
    let mx_correction = OPENGL_TO_WGPU_MATRIX;
    mx_correction * mx_projection * mx_view
}

fn main() {
    use wgpu::winit::{
        ElementState, Event, EventsLoop, KeyboardInput, VirtualKeyCode, WindowEvent,
    };

    env_logger::init();

    let mut events_loop = EventsLoop::new();

    info!("Initializing the window...");

    #[cfg(not(feature = "gl"))]
    let (_window, instance, hidpi_factor, size, surface) = {
        use wgpu::winit::Window;

        let instance = wgpu::Instance::new();

        let window = Window::new(&events_loop).unwrap();
        window.set_title("Test");
        let hidpi_factor = window.get_hidpi_factor();
        let size = window.get_inner_size().unwrap().to_physical(hidpi_factor);

        let surface = instance.create_surface(&window);

        (window, instance, hidpi_factor, size, surface)
    };

    #[cfg(feature = "gl")]
    let (instance, hidpi_factor, size, surface) = {
        let wb = wgpu::winit::WindowBuilder::new();
        let cb = wgpu::glutin::ContextBuilder::new().with_vsync(true);
        let context = wgpu::glutin::WindowedContext::new_windowed(wb, cb, &events_loop).unwrap();
        context.window().set_title(title);

        let hidpi_factor = context.window().get_hidpi_factor();
        let size = context
            .window()
            .get_inner_size()
            .unwrap()
            .to_physical(hidpi_factor);

        let instance = wgpu::Instance::new(context);
        let surface = instance.get_surface();

        (instance, hidpi_factor, size, surface)
    };

    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::LowPower,
    });

    let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    });

    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8Unorm,
        width: size.width.round() as u32,
        height: size.height.round() as u32,
        present_mode: wgpu::PresentMode::Vsync,
    };
    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);

    info!("Initializing the example...");

    // Create the vertex and index buffers
    let vertex_size = mem::size_of::<Vertex>();
    let (vertex_data, index_data) = create_vertices();
    let vertex_buf = device
        .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&vertex_data);

    let index_buf = device
        .create_buffer_mapped(index_data.len(), wgpu::BufferUsage::INDEX)
        .fill_from_slice(&index_data);

    let mut instance_data = Vec::new();
    let instance_size = mem::size_of::<Instance>();
    let count = 64;
    for i in 0..(count * count * count) {
        instance_data.push(Instance {
            _pos: [
                -1.0 + 2.0 * ((i / (count * count)) as f32) / count as f32,
                -1.0 + 2.0 * (((i / count) % count) as f32) / count as f32,
                -1.0 + 2.0 * ((i % count) as f32) / count as f32,
                0.0,
            ],
            _size: [0.05, 0.05, 0.05, 0.0],
        });
    }
    let instance_buf = device
        .create_buffer_mapped(instance_data.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&instance_data);

    let big_instance_data = vec![
        Instance {
            _pos: [-2.0, 0.0, 0.0, 0.0],
            _size: [0.2, 1.0, 1.0, 0.0],
        },
        Instance {
            _pos: [2.0, 0.0, 0.0, 0.0],
            _size: [0.2, 1.0, 1.0, 0.0],
        },
    ];;
    let big_instance_buf = device
        .create_buffer_mapped(big_instance_data.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(&big_instance_data);

    // Create pipeline layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutBinding {
            binding: 0,
            visibility: wgpu::ShaderStage::VERTEX,
            ty: wgpu::BindingType::UniformBuffer,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let mx_total = generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
    let mx_ref: &[f32; 16] = mx_total.as_ref();
    let uniform_buf = device
        .create_buffer_mapped(
            16,
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
        )
        .fill_from_slice(mx_ref);

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &uniform_buf,
                range: 0..64,
            },
        }],
    });

    // Create the render pipeline
    let vs_bytes = load_glsl(include_str!("shader.vert"), ShaderStage::Vertex);
    let fs_bytes = load_glsl(include_str!("shader.frag"), ShaderStage::Fragment);
    let cs_bytes = load_glsl(include_str!("shader.comp"), ShaderStage::Compute);
    let vs_module = device.create_shader_module(&vs_bytes);
    let fs_module = device.create_shader_module(&fs_bytes);
    let cs_module = device.create_shader_module(&cs_bytes);

    let cs_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[],
    });
    let cs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &cs_pipeline_layout,
        compute_stage: wgpu::PipelineStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::PipelineStageDescriptor {
            module: &vs_module,
            entry_point: "main",
        },
        fragment_stage: Some(wgpu::PipelineStageDescriptor {
            module: &fs_module,
            entry_point: "main",
        }),
        rasterization_state: wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::Back,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        },
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: sc_desc.format,
            color_blend: wgpu::BlendDescriptor::REPLACE,
            alpha_blend: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWrite::ALL,
        }],
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[
            wgpu::VertexBufferDescriptor {
                stride: vertex_size as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float2,
                        offset: 4 * 4,
                        shader_location: 1,
                    },
                ],
            },
            wgpu::VertexBufferDescriptor {
                stride: instance_size as wgpu::BufferAddress,
                step_mode: wgpu::InputStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float4,
                        offset: 0,
                        shader_location: 2,
                    },
                    wgpu::VertexAttributeDescriptor {
                        format: wgpu::VertexFormat::Float4,
                        offset: 4 * 4,
                        shader_location: 3,
                    },
                ],
            },
        ],
        sample_count: 1,
    });

    info!("Entering render loop...");
    let mut running = true;
    while running {
        let frame = swap_chain.get_next_texture();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&cs_pipeline);
            cpass.dispatch(1, 1, 1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    },
                }],
                depth_stencil_attachment: None,
            });
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_index_buffer(&index_buf, 0);
            rpass.set_vertex_buffers(&[(&vertex_buf, 0), (&instance_buf, 0)]);
            rpass.draw_indexed(0..index_data.len() as u32, 0, 0..instance_data.len() as u32);
            rpass.set_vertex_buffers(&[(&vertex_buf, 0), (&big_instance_buf, 0)]);
            rpass.draw_indexed(0..index_data.len() as u32, 0, 0..big_instance_data.len() as u32);
        }

        device.get_queue().submit(&[encoder.finish()]);
        running &= !cfg!(feature = "metal-auto-capture");
    }
}
