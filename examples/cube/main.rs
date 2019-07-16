#[path = "../framework.rs"]
mod framework;

#[derive(Clone, Copy)]
struct Vertex {
    _pos: [f32; 3],
    _tex_coord: [f32; 2],
}

#[derive(Clone, Copy)]
struct Instance {
    _pos: [f32; 3],
}

#[derive(Clone, Copy)]
struct DrawArguments {
    index_count: u32,
    instance_count: u32,
    base_index: u32,
    vertex_offset: i32,
    base_instance: u32,
}

#[repr(C)]
struct OcclusionUniforms {
    proj: [[f32; 4]; 4],
}

struct Pass {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
}

fn create_depth_texture(device: &mut wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: width,
            height: height,
            depth: 1,
        },
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::D32Float,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
    });
    depth_texture.create_default_view()
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
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

fn create_instances() -> Vec<Instance> {
    vec![
        Instance {
            _pos: [0.0 as f32, 0.0, -2.0],
        },
        Instance {
            _pos: [2.5 as f32, 0.0, -2.0],
        },
        Instance {
            _pos: [0.0 as f32, 2.5, -2.0],
        },
        Instance {
            _pos: [2.5 as f32, 2.5, -2.0],
        },
        Instance {
            _pos: [-2.5 as f32, 0.0, -2.0],
        },
    ]
}

fn create_texels(size: usize) -> Vec<u8> {
    use std::iter;

    (0 .. size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let cx = 3.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let cy = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let (mut x, mut y, mut count) = (cx, cy, 0);
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 5) as u8)
                .chain(iter::once(0xFF - (count * 15) as u8))
                .chain(iter::once(0xFF - (count * 50) as u8))
                .chain(iter::once(1))
        })
        .collect()
}

fn create_plane(size: i8) -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        vertex([size, -size, 0], [0, 0]),
        vertex([size, size, 0], [0, 1]),
        vertex([-size, -size, 0], [1, 1]),
        vertex([-size, size, 0], [1, 0]),
    ];

    let index_data: &[u16] = &[0, 1, 2, 2, 1, 3];

    (vertex_data.to_vec(), index_data.to_vec())
}


struct Example {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: usize,
    draw_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    draw_data: Vec<DrawArguments>,
    frame_id: usize,
    instance_buf: wgpu::Buffer,
    instance_count: usize,
    depth_texture: wgpu::TextureView,
    occlusion_pass: Pass,
    occlusion_texture: wgpu::TextureView,
    visibility_buf: wgpu::Buffer,
    visibility_data_size: wgpu::BufferAddress,
}

impl Example {
    const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::D32Float;
    const OCCLUSION_SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth: 1,
    };

    fn generate_matrix(aspect_ratio: f32) -> cgmath::Matrix4<f32> {
        let mx_projection = cgmath::perspective(cgmath::Deg(45f32), aspect_ratio, 1.0, 100.0);
        let mx_view = cgmath::Matrix4::look_at(
            cgmath::Point3::new(5.0f32, -8.0, 1.0),
            cgmath::Point3::new(0f32, 0.0, 0.0),
            cgmath::Vector3::unit_z(),
        );
        let mx_correction = framework::OPENGL_TO_WGPU_MATRIX;
        mx_correction * mx_projection * mx_view
    }
}

impl framework::Example for Example {
    fn init(sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) -> Self {
        use std::mem;

        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        // Create the vertex and index buffers
        let vertex_size = mem::size_of::<Vertex>();
        let (vertex_data, index_data) = create_vertices();
        let vertex_buf = device
            .create_buffer_mapped(vertex_data.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(&vertex_data);

        let index_buf = device
            .create_buffer_mapped(index_data.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(&index_data);

        let instance_size = mem::size_of::<Instance>();
        let instance_data = create_instances();
        let instance_buf = device
            .create_buffer_mapped(instance_data.len(), wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::STORAGE)
            .fill_from_slice(&instance_data);

        let visibility_data = vec![0 as u32, 0, 0, 0, 0];
        let visibility_buf = device
            .create_buffer_mapped(visibility_data.len(), wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::TRANSFER_SRC)
            .fill_from_slice(&visibility_data);

        let (plane_vertex_data, plane_index_data) = create_plane(7);
        let plane_vertex_buf = device
            .create_buffer_mapped(plane_vertex_data.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(&plane_vertex_data);

        let plane_index_buf = device
            .create_buffer_mapped(plane_index_data.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(&plane_index_data);

        //let local_bind_group_layout =
            //device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                //bindings: &[wgpu::BindGroupLayoutBinding {
                    //binding: 0,
                    //visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
                    //ty: wgpu::BindingType::UniformBuffer,
                //}],
            //});

        let draw_data = vec![DrawArguments{
            index_count: index_data.len() as u32,
            instance_count: 1,
            base_index: 0,
            vertex_offset: 0,
            base_instance: 0,
        }];

        let draw_buf = device
            .create_buffer(&wgpu::BufferDescriptor {
                size: (draw_data.len() * std::mem::size_of::<DrawArguments>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsage::INDIRECT | wgpu::BufferUsage::STORAGE
            });

        // Create pipeline layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::SampledTexture,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStage::FRAGMENT | wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::Sampler,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        // Create the texture
        //let size = 256u32;
        //let texels = create_texels(size as usize);
        //let texture_extent = wgpu::Extent3d {
            //width: size,
            //height: size,
            //depth: 1,
        //};
        //let texture = device.create_texture(&wgpu::TextureDescriptor {
            //size: texture_extent,
            //array_layer_count: 1,
            //mip_level_count: 1,
            //sample_count: 1,
            //dimension: wgpu::TextureDimension::D2,
            //format: wgpu::TextureFormat::Rgba8Unorm,
            //usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::TRANSFER_DST,
        //});
        //let texture_view = texture.create_default_view();
        //let temp_buf = device
            //.create_buffer_mapped(texels.len(), wgpu::BufferUsage::TRANSFER_SRC)
            //.fill_from_slice(&texels);
        //init_encoder.copy_buffer_to_texture(
            //wgpu::BufferCopyView {
                //buffer: &temp_buf,
                //offset: 0,
                //row_pitch: 4 * size,
                //image_height: size,
            //},
            //wgpu::TextureCopyView {
                //texture: &texture,
                //mip_level: 0,
                //array_layer: 0,
                //origin: wgpu::Origin3d {
                    //x: 0.0,
                    //y: 0.0,
                    //z: 0.0,
                //},
            //},
            //texture_extent,
        //);

        // Occlusion texture
        let occlusion_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::LessEqual,
        });

        let occlusion_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: Self::OCCLUSION_SIZE,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::OCCLUSION_FORMAT,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::SAMPLED,
        });
        let occlusion_default_view = occlusion_texture.create_default_view();
        let occlusion_view = occlusion_texture.create_view(&wgpu::TextureViewDescriptor {
            format: Self::OCCLUSION_FORMAT,
            dimension: wgpu::TextureViewDimension::D2,
            aspect: wgpu::TextureAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            array_count: 1,
        });

        // Create other resources
        //let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            //address_mode_u: wgpu::AddressMode::ClampToEdge,
            //address_mode_v: wgpu::AddressMode::ClampToEdge,
            //address_mode_w: wgpu::AddressMode::ClampToEdge,
            //mag_filter: wgpu::FilterMode::Nearest,
            //min_filter: wgpu::FilterMode::Linear,
            //mipmap_filter: wgpu::FilterMode::Nearest,
            //lod_min_clamp: -100.0,
            //lod_max_clamp: 100.0,
            //compare_function: wgpu::CompareFunction::Always,
        //});
        let mx_total = Self::generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
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
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0 .. 64,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&occlusion_default_view),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&occlusion_sampler),
                },
            ],
        });

        // Create the render pipeline
        let vs_bytes =
            framework::load_glsl(include_str!("shader.vert"), framework::ShaderStage::Vertex);
        let fs_bytes = framework::load_glsl(
            include_str!("shader.frag"),
            framework::ShaderStage::Fragment,
        );
        let vs_module = device.create_shader_module(&vs_bytes);
        let fs_module = device.create_shader_module(&fs_bytes);


        let vb_desc = [wgpu::VertexBufferDescriptor {
            stride: vertex_size as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float2,
                    offset: 4 * 3,
                    shader_location: 1,
                },
            ],
        },
        wgpu::VertexBufferDescriptor {
            stride: instance_size as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Instance,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    format: wgpu::VertexFormat::Float3,
                    offset: 0,
                    shader_location: 2,
                },
            ],
        }];

        let occlusion_pass = {
            // Create pipeline layout
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    bindings: &[wgpu::BindGroupLayoutBinding {
                        binding: 0, // global
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::UniformBuffer,
                    }],
                });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let uniform_size = mem::size_of::<OcclusionUniforms>() as wgpu::BufferAddress;

            let uniform_buf = device
                .create_buffer_mapped(
                    16,
                    wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
                )
                .fill_from_slice(mx_ref);
            //let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
                //size: uniform_size,
                //usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::TRANSFER_DST,
            //});

            // Create bind group
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buf,
                        range: 0 .. uniform_size,
                    },
                }],
            });

            // Create the render pipeline
            let vs_bytes =
                framework::load_glsl(include_str!("occlusion.vert"), framework::ShaderStage::Vertex);
            let fs_bytes =
                framework::load_glsl(include_str!("occlusion.frag"), framework::ShaderStage::Fragment);
            let vs_module = device.create_shader_module(&vs_bytes);
            let fs_module = device.create_shader_module(&fs_bytes);

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
                    depth_bias: 0, // corresponds to bilinear filtering
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                },
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                color_states: &[],
                depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                    format: Self::OCCLUSION_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    stencil_read_mask: 0,
                    stencil_write_mask: 0,
                }),
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &vb_desc,
                sample_count: 1,
            });

            Pass {
                pipeline,
                bind_group,
                uniform_buf,
            }
        };

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
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::D32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
            }),
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &vb_desc,
            sample_count: 1,
        });

        // Compute stuff
        let cs_bytes =
            framework::load_glsl(include_str!("shader.comp"), framework::ShaderStage::Compute);
        let cs_module = device.create_shader_module(&cs_bytes);

        let draw_data_size = (draw_data.len() * std::mem::size_of::<DrawArguments>()) as wgpu::BufferAddress;

        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer,
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 2,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer,
                },
            ],
        });

        let instance_data_size = (instance_data.len() * instance_size) as wgpu::BufferAddress;
        let visibility_data_size = (visibility_data.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &draw_buf,
                        range: 0 .. draw_data_size,
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &instance_buf,
                        range: 0 .. instance_data_size,
                    },
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &visibility_buf,
                        range: 0 .. visibility_data_size,
                    },
                },
            ],

        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&compute_bind_group_layout, &bind_group_layout],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            layout: &compute_pipeline_layout,
            compute_stage: wgpu::PipelineStageDescriptor {
                module: &cs_module,
                entry_point: "main",
            },
        });

        let depth_texture = create_depth_texture(device, sc_desc.width, sc_desc.height);

        // Done
        let init_command_buf = init_encoder.finish();
        device.get_queue().submit(&[init_command_buf]);
        Example {
            vertex_buf,
            index_buf,
            index_count: index_data.len(),
            draw_buf,
            bind_group,
            uniform_buf,
            pipeline,
            compute_pipeline,
            compute_bind_group,
            draw_data,
            frame_id: 0,
            instance_buf,
            instance_count: instance_data.len(),
            depth_texture,
            occlusion_pass,
            occlusion_texture: occlusion_view,
            visibility_buf,
            visibility_data_size,
        }
    }

    fn update(&mut self, _event: wgpu::winit::WindowEvent) {
        //empty
    }

    fn resize(&mut self, sc_desc: &wgpu::SwapChainDescriptor, device: &mut wgpu::Device) {
        let mx_total = Self::generate_matrix(sc_desc.width as f32 / sc_desc.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();

        let temp_buf = device
            .create_buffer_mapped(16, wgpu::BufferUsage::TRANSFER_SRC)
            .fill_from_slice(mx_ref);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
        encoder.copy_buffer_to_buffer(&temp_buf, 0, &self.uniform_buf, 0, 64);
        device.get_queue().submit(&[encoder.finish()]);
        self.depth_texture = create_depth_texture(device, sc_desc.width, sc_desc.height);
    }

    fn render(&mut self, frame: &wgpu::SwapChainOutput, device: &mut wgpu::Device) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &self.occlusion_texture,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });
            pass.set_pipeline(&self.occlusion_pass.pipeline);
            pass.set_bind_group(0, &self.occlusion_pass.bind_group, &[]);

            // TODO replace with occluders
            pass.set_index_buffer(&self.index_buf, 0);
            pass.set_vertex_buffers(&[(&self.vertex_buf, 0), (&self.instance_buf, 0)]);
            pass.draw_indexed(0 .. self.index_count as u32, 0, 0 .. 5);
        }

        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bind_group, &[]);
            cpass.set_bind_group(1, &self.bind_group, &[]);
            //cpass.dispatch(self.draw_data.len() as u32, 1, 1);
            cpass.dispatch(self.instance_count as u32, 1, 1);
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                        attachment: &self.depth_texture,
                        depth_load_op: wgpu::LoadOp::Clear,
                        depth_store_op: wgpu::StoreOp::Store,
                        stencil_load_op: wgpu::LoadOp::Clear,
                        stencil_store_op: wgpu::StoreOp::Store,
                        clear_depth: 1.0,
                        clear_stencil: 0,
                }),
            });
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.set_index_buffer(&self.index_buf, 0);
            rpass.set_vertex_buffers(&[(&self.vertex_buf, 0), (&self.instance_buf, 0)]);
            //rpass.draw_indexed(0 .. self.index_count as u32, 0, 0 .. 5);
            rpass.draw_indexed_indirect(&self.draw_buf, 0);
        }
        let temp_buf = device
            .create_buffer(&wgpu::BufferDescriptor {
                size: self.visibility_data_size,
                usage: wgpu::BufferUsage::TRANSFER_DST | wgpu::BufferUsage::MAP_READ
            });
        encoder.copy_buffer_to_buffer(&self.visibility_buf, 0, &temp_buf, 0, self.visibility_data_size);

        self.frame_id += 1;

        device.get_queue().submit(&[encoder.finish()]);

        temp_buf.map_read_async(0, self.visibility_data_size, |result: wgpu::BufferMapAsyncResult<&[u32]>| {
            if let Ok(mapping) = result {
                println!("Times: {:?}", mapping.data);
            }
        });
    }
}

fn main() {
    framework::run::<Example>("cube");
}
