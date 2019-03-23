use image::{ImageBuffer, Rgba};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use vulkano::command_buffer::{CommandBuffer, DynamicState, AutoCommandBufferBuilder};
use vulkano::sync::GpuFuture;
use std::sync::Arc;
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, ComputePipeline};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::format::{ClearValue, Format};
use vulkano::image::{StorageImage, Dimensions};
use vulkano::framebuffer::{Framebuffer, Subpass};

struct Vertex { position: [f32; 2] } 
vulkano::impl_vertex!(Vertex, position);

fn main() {
   
    /* We create a Vulkano instance, which lets use use the underlying
     * Vulkan API. */
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("Failed to create new instance.");
   
    /* There could be many devices that support Vulkan. For instance, a video card or an
     * integrated graphics unit. We need to select which one we want to use. Note: This
     * would probably be a decision best made by the user. */
    let physical = PhysicalDevice::enumerate(&instance).next().expect("No device available.");

    /* Every device that supports Vulkan is issued commands through queues. Queues are
     * grouped by queue families, and some families support more than one queue. Some
     * families only support a specific type of operations, like compute or rendering.*/
    let queue_family = physical.queue_families().find(|&q| q.supports_graphics()) 
        .expect("Could not find a graphical queue family");
   
    /* Now we can create the device object. This will return the device itself along with
     * a list of queue objects that we can use to submit operations. */
    let (device, mut queues) = {
        Device::new(physical, &Features::none(), &DeviceExtensions::none(),
                    [(queue_family, 0.5)].iter().cloned()).expect("Failed to create device")
    };
    let queue = queues.next().unwrap();
   
    /* We share memory with devices through buffers. Different buffers are optimized for 
     * different things. For example, there are ImmutableBuffers and CpuBufferPools. 
     * We specify the device this buffer will communicate with, since device is Arc<Device>,
     * this will not be expensive. We can also give hints to the implementation using
     * BufferUsage. Here we allow all types of use. */
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), 0..64)
        .unwrap();
    let dest =   CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0..64).map(|_| 0u8))
        .unwrap();
    
    /* We send commands to the GPU by using a command buffer. The AutoCommandBufferBuilder struct
     * allows us to easily build command buffers to be sent. */
    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .copy_buffer(source.clone(), dest.clone()).unwrap()
        .build().unwrap();

    /* We send the command down our queue and get the result in finished. We must wait for the
     * results to be written back, so we fence, flush our caches, and then wait for the GPU
     * to finish executing the command. */
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    /* Source and Dest are RwLocks, so reading them gives us an immutable reference.*/
    let src_content = source.read().unwrap();
    let dst_content = dest.read().unwrap();

    assert_eq!(&*src_content, &*dst_content);

    /* We will now perform an arbitrary operation using a compute shader. We will multiply each
     * element of this buffer by 12. */
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), 0..65536)
        .unwrap();

    let shader = cs::Shader::load(device.clone()).expect("Failed to create shader module");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
        .expect("failed to create compute pipeline"));
   
    let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([1024,1,1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .build().unwrap();

    command_buffer.execute(queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    
    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }


    println!("Success");

    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 512, height: 512 },
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0..512*512*4).map(|_| 0u8))
        .expect("Failed to create buffer");

    let shader = mandelbrot::Shader::load(device.clone()).expect("Could not load mandelbrot shader");

    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap());


    let set = Arc::new(PersistentDescriptorSet::start(compute_pipeline.clone(), 0)
        .add_image(image.clone()).unwrap()
        .build().unwrap());

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family()).unwrap()
        .dispatch([512 / 8, 512 / 8, 1], compute_pipeline.clone(), set.clone(), ()).unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone()).unwrap()
        .build().unwrap();

    command_buffer.execute(queue.clone()).unwrap().then_signal_fence_and_flush().unwrap().wait(None).unwrap();
    
    let buffer_content = buf.read().unwrap();
    let mand = ImageBuffer::<Rgba<u8>, _>::from_raw(512, 512, &buffer_content[..]).unwrap();
    mand.save("mandelbor.png").unwrap();

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), (0..512*512*4).map(|_| 0u8))
        .expect("Failed to create buffer");

    /* Render a triangle! */
    let v1 = Vertex { position: [-0.5, -0.5] };
    let v2 = Vertex { position: [0.0, 0.5]   };
    let v3 = Vertex { position: [0.5, -0.25] };
    
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        vec![v1, v2, v3].into_iter()).unwrap();

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: Format::R8G8B8A8Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    ).unwrap());

    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(image.clone()).unwrap()
        .build().unwrap());

    let vs = vertex::Shader::load(device.clone()).expect("Failed to create vertex shader");
    let fs = frag::Shader::load(device.clone()).expect("Failed to create fragment shader");

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0,0.0],
            dimensions: [512.0, 512.0],
            depth_range: 0.0 .. 1.0,
        }]),
        .. DynamicState::none()
    };

    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(
        device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0,0.0,0.0,0.0].into()])
        .unwrap()
        .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
        .unwrap()
        .end_render_pass()
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()
        .build()
        .unwrap();

    command_buffer.execute(queue.clone()).unwrap()
        .then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(512, 512, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();
}

mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/op.glsl"
    }
}

mod mandelbrot {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/mandelbrot.glsl"
    }
}

mod vertex {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/vertex.glsl"
    }
}

mod frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/frag.glsl"
    }
}
