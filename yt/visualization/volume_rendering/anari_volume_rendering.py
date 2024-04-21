import numpy as np

from yt.funcs import is_sequence, mylog
from yt.utilities.lib.image_samplers import (
    ImageSampler,
)
from anari import *

anari_library_name = "ospray"
anari_device_subtype = "default"
anari_renderer_name = "default"

prefixes = {
    lib.ANARI_SEVERITY_FATAL_ERROR : "FATAL",
    lib.ANARI_SEVERITY_ERROR : "ERROR",
    lib.ANARI_SEVERITY_WARNING : "WARNING",
    lib.ANARI_SEVERITY_PERFORMANCE_WARNING : "PERFORMANCE",
    lib.ANARI_SEVERITY_INFO : "INFO",
    lib.ANARI_SEVERITY_DEBUG : "DEBUG"
}

renderer_enum_info = [("default", "default", "default")]

def anari_status(device, source, sourceType, severity, code, message):
    print('[%s]: '%prefixes[severity]+message)

def get_renderer_enum_info(self, context):
    global renderer_enum_info
    return renderer_enum_info

status_handle = ffi.new_handle(anari_status) #something needs to keep this handle alive

class AnariSampler:

    def __init__(self, camera, render_source, params):
        self.params = params

        self.init_image_sampler(camera, render_source)
        self.aimage = self.image_sampler.aimage

        # ANARI library
        self.library = anariLoadLibrary(anari_library_name, status_handle)
        if not self.library:
            print("Error loading ANARI library")
            return

        # ANARI device
        self.device = anariNewDevice(self.library, anari_device_subtype)
        if not self.device:
            print("ANARI device not initialized")
            return

        # ANARI renderer
        self.renderer = anariNewRenderer(self.device, anari_renderer_name)
        anariSetParameter(self.device, self.renderer, "ambientColor", ANARI_FLOAT32_VEC3, [1,1,1])
        anariSetParameter(self.device, self.renderer, "ambientRadiance", ANARI_FLOAT32, 1)
        anariSetParameter(self.device, self.renderer, "pixelSamples", ANARI_INT32, 8)
        anariCommitParameters(self.device, self.renderer)

        cam_pos = [ camera.position[0],
                    camera.position[1],
                    camera.position[2] ]

        cam_dir = [ camera.focus[0]-camera.position[0],
                    camera.focus[1]-camera.position[1],
                    camera.focus[2]-camera.position[2] ]

        cam_up = [ camera.north_vector[0],
                   camera.north_vector[1],
                   camera.north_vector[2] ]

        # ANARI camera
        if self.params["lens_type"] == 'plane-parallel':
            self.camera = anariNewCamera(self.device, 'orthographic')
            anariSetParameter(self.device, self.camera, 'position', ANARI_FLOAT32_VEC3, cam_pos)
            anariSetParameter(self.device, self.camera, 'direction', ANARI_FLOAT32_VEC3, cam_dir)
            anariSetParameter(self.device, self.camera, 'up', ANARI_FLOAT32_VEC3, cam_up)
            anariSetParameter(self.device, self.camera, 'height', ANARI_FLOAT32, camera.width[1])
        elif self.params["lens_type"] == 'perspective':
            self.camera = anariNewCamera(self.device, 'perspective')
            anariSetParameter(self.device, self.camera, 'position', ANARI_FLOAT32_VEC3, cam_pos)
            anariSetParameter(self.device, self.camera, 'direction', ANARI_FLOAT32_VEC3, cam_dir)
            anariSetParameter(self.device, self.camera, 'up', ANARI_FLOAT32_VEC3, cam_up)
            # ...

        if self.camera is None:
            print("Camera initialization failed")
            return

        anariCommitParameters(self.device, self.camera)

        # ANARI frame
        self.frame = anariNewFrame(self.device)
        anariSetParameter(self.device, self.frame, 'channel.color', ANARI_DATA_TYPE, ANARI_UFIXED8_VEC4)
        anariSetParameter(self.device, self.frame, 'channel.depth', ANARI_DATA_TYPE, ANARI_FLOAT32)

        width = self.aimage.shape[0]
        height = self.aimage.shape[1]
        anariSetParameter(self.device, self.frame, 'size', ANARI_UINT32_VEC2, [width, height])

        anariSetParameter(self.device, self.frame, 'renderer', ANARI_OBJECT, self.renderer)
        anariSetParameter(self.device, self.frame, 'camera', ANARI_OBJECT, self.camera)

        anariCommitParameters(self.device, self.frame)

    def render_frame(self, volume, tf, viewpoint):
        if not self.frame:
            print("render_frame: Frame invalid..")
            return

        # TODO: should probably do this on init?!
        volume.create_world(self.device, tf)
        anariSetParameter(self.device, self.frame, 'world', ANARI_OBJECT, volume.world)
        anariCommitParameters(self.device, self.frame)

        anariRenderFrame(self.device, self.frame)
        anariFrameReady(self.device, self.frame, ANARI_WAIT)

        width = self.aimage.shape[0]
        height = self.aimage.shape[1]

        void_pixels, frame_width, frame_height, frame_type = anariMapFrame(self.device, self.frame, 'channel.color')

        if frame_width != width or frame_height != height:
            print('something went wrong..')
            return

        unpacked_pixels = ffi.buffer(void_pixels, frame_width*frame_height*4)
        pixels = np.array(unpacked_pixels).astype(np.float32)#*(1.0/255.0)
        rect = pixels.reshape((width, height, 4))
        anariUnmapFrame(self.device, self.frame, 'channel.color')

        for x in range(width):
            for y in range(height):
                for c in range(4):
                    self.aimage[x, y, c] = rect[x, y, c]

    def init_image_sampler(self, camera, render_source):
        args = (
            np.atleast_3d(self.params["vp_pos"]),
            np.atleast_3d(self.params["vp_dir"]),
            self.params["center"],
            self.params["bounds"],
            self.params["image"],
            self.params["x_vec"],
            self.params["y_vec"],
            self.params["width"],
            render_source.volume_method,
            self.params["transfer_function"],
            self.params["num_samples"],
        )
        kwargs = {
            "lens_type": self.params["lens_type"],
        }
        if render_source.zbuffer is not None:
            kwargs["zbuffer"] = render_source.zbuffer.z
            args[4][:] = np.reshape(
                render_source.zbuffer.rgba[:],
                (camera.resolution[0], camera.resolution[1], 4),
            )
        else:
            kwargs["zbuffer"] = np.ones(self.params["image"].shape[:2], "float64")

        self.image_sampler = ImageSampler(*args, **kwargs)


class AnariVolume:

    fields = None
    log_fields = None

    def __init__(self, ds, data_source):
        self.ds = ds
        self.data_source = data_source

        self.min_level = 0
        self.max_level = ds.index.max_level

        if data_source is None:
            data_source = ds.all_data()

    def set_fields(self, fields, log_fields, no_ghost, force=False):
        new_fields = self.data_source._determine_fields(fields)
        regenerate_data = (
            self.fields is None
            or len(self.fields) != len(new_fields)
            or self.fields != new_fields
            or force
        )
        if not is_sequence(log_fields):
            log_fields = [log_fields]
        new_log_fields = list(log_fields)
        self.fields = new_fields

        if self.log_fields is not None and not regenerate_data:
            flip_log = list(map(operator.ne, self.log_fields, new_log_fields))
        else:
            flip_log = [False] * len(new_log_fields)
        self.log_fields = new_log_fields

    def tf_to_rgb(self, tf, height, width):
        rgb = np.zeros((height, width, 3), dtype="float32")
        hvals = np.mgrid[tf.x_bounds[0] : tf.x_bounds[1] : height * 1j]
        for i, f in enumerate(tf.funcs[:3]):
            vals = np.interp(hvals, f.x, f.y)
            rgb[:, :, i] = (vals[:, None] * 1.0).astype("float32")
        rgb = rgb[::-1, :, :]
        return rgb

    def tf_to_alpha(self, tf, height, width):
        alpha = np.zeros((height, width, 1), dtype="float32")
        hvals = np.mgrid[tf.x_bounds[0] : tf.x_bounds[1] : height * 1j]
        vals = np.interp(hvals, tf.alpha.x, tf.alpha.y)
        alpha[:, :, 0] = (vals[:, None] * 1.0).astype("float32")
        alpha = alpha[::-1, :, :]
        return alpha

    def set_array(self, device, obj, name, atype, count, arr):
        (ptr, stride) = anariMapParameterArray1D(device, obj, name, atype, count)
        ffi.memmove(ptr, ffi.from_buffer(arr), arr.size*arr.itemsize)
        anariUnmapParameterArray(device, obj, name)

    def create_world(self, device, tf):
        self.world = anariNewWorld(device)
        self.volume = anariNewVolume(device, 'transferFunction1D')
        self.field = anariNewSpatialField(device, 'amr')

        cellWidth = []
        blockBounds = []
        blockLevel = []
        blockData = []

        lvl_range = range(self.min_level, self.max_level + 1)

        # cell bounds (in logical coords):
        x_min = 10000000
        y_min = 10000000
        z_min = 10000000

        x_max = 0
        y_max = 0
        z_max = 0

        # eventually, according to the provisional specs, level-0
        # will become the coarsest, but he devices out there who
        # support amr (visionaray, ospray) still have this the
        # other way around:
        anari_level_0_is_finest = True

        for level in lvl_range:
            cw = 0.0
            if anari_level_0_is_finest:
                cw = 2**(self.max_level - level)
            else:
                cw = 2**level

            cellWidth.append(cw)

            grids = np.array(
                [b for b, mask in self.data_source.blocks if b.Level == level]
            )
            if len(grids) == 0:
                continue

            # cell bounds (in logical coords) on this level:
            level_x_min = 10000000
            level_y_min = 10000000
            level_z_min = 10000000

            level_x_max = 0
            level_y_max = 0
            level_z_max = 0

            for grid in grids:
                lower = np.rint(grid.LeftEdge / grid.dds).astype("int32")
                upper = np.rint(grid.RightEdge / grid.dds).astype("int32")

                box_min = [ lower[0]*cw,
                            lower[1]*cw,
                            lower[2]*cw ]

                box_max = [ upper[0]*cw,
                            upper[1]*cw,
                            upper[2]*cw ]

                x_min = min(x_min, box_min[0])
                y_min = min(y_min, box_min[1])
                z_min = min(z_min, box_min[2])

                x_max = max(x_max, box_max[0])
                y_max = max(y_max, box_max[1])
                z_max = max(z_max, box_max[2])

                #print("level: ",level)
                #print("box_min,box_max: ",box_min,box_max)
                #print("lower,upper: ",lower,upper)

                level_x_min = min(level_x_min, box_min[0])
                level_y_min = min(level_y_min, box_min[1])
                level_z_min = min(level_z_min, box_min[2])

                level_x_max = max(level_x_max, box_max[0])
                level_y_max = max(level_y_max, box_max[1])
                level_z_max = max(level_z_max, box_max[2])

                dims = upper - lower
                blockBounds.append([lower, upper])
                blockLevel.append(level)
                data = np.asarray(grid[self.fields[0]], dtype=np.float32)
                flat = np.zeros(dims[0] * dims[1] * dims[2], dtype=np.float32)
                for x, l in enumerate(data):
                    for y, ll in enumerate(l):
                        for z, e in enumerate(ll):
                            index = x+y*dims[0]+z*dims[0]*dims[1]
                            if self.log_fields:
                                flat[index] = np.log10(e)
                            else:
                                flat[index] = e
                blockData.append(np.asarray(flat, dtype=np.float32))
            #print("level: ",level)
            #print("bounds:")
            #print(level_x_min, level_y_min, level_z_min)
            #print(level_x_max, level_y_max, level_z_max)

        #print(x_min, y_min, z_min)
        #print(x_max, y_max, z_max)

        vmin = tf.x_bounds[0]
        vmax = tf.x_bounds[1]

        levelcount = len(cellWidth)
        blockcount = len(blockLevel)
        npcellWidth = np.zeros([levelcount], dtype=np.float32)
        npblockBounds = np.zeros([blockcount*6], dtype=np.int32)
        npblockLevel = np.zeros([blockcount], dtype=np.int32)
        dataArrays = []

        for i in range(levelcount):
            npcellWidth[i] = cellWidth[i]

        for i in range(blockcount):
            npblockBounds[i*6]   = blockBounds[i][0][0]
            npblockBounds[i*6+1] = blockBounds[i][0][1]
            npblockBounds[i*6+2] = blockBounds[i][0][2]
            npblockBounds[i*6+3] = blockBounds[i][1][0]-1
            npblockBounds[i*6+4] = blockBounds[i][1][1]-1
            npblockBounds[i*6+5] = blockBounds[i][1][2]-1
            npblockLevel[i] = blockLevel[i]
            width = blockBounds[i][1][0]-blockBounds[i][0][0]
            height = blockBounds[i][1][1]-blockBounds[i][0][1]
            depth = blockBounds[i][1][2]-blockBounds[i][0][2]
            bd = anariNewArray3D(device, ffi.from_buffer(blockData[i]), ANARI_FLOAT32, width, height, depth)
            dataArrays.append(bd)

        npblockData = ffi.new('ANARIArray3D[]', dataArrays)
        array = anariNewArray1D(device, npblockData, ANARI_ARRAY1D, len(npblockData))

        # compute gridOrigin and gridSpacing:
        domain_width = self.data_source.ds.domain_width
        domain_center = self.data_source.ds.domain_center

        gridOrigin = [ domain_center[0] - domain_width[0]*0.5,
                       domain_center[1] - domain_width[1]*0.5,
                       domain_center[2] - domain_width[2]*0.5 ]

        gridSpacing = [ domain_width[0] / (x_max-x_min),
                        domain_width[1] / (y_max-y_min),
                        domain_width[2] / (z_max-z_min) ]

        self.set_array(device, self.field, 'cellWidth', ANARI_FLOAT32, levelcount, npcellWidth)
        self.set_array(device, self.field, 'block.bounds', ANARI_INT32_BOX3, blockcount, npblockBounds)
        self.set_array(device, self.field, 'block.level', ANARI_INT32, blockcount, npblockLevel)
        anariSetParameter(device, self.field, 'block.data', ANARI_ARRAY3D, array)
        anariSetParameter(device, self.field, 'gridOrigin', ANARI_FLOAT32_VEC3, gridOrigin)
        anariSetParameter(device, self.field, 'gridSpacing', ANARI_FLOAT32_VEC3, gridSpacing)

        anariCommitParameters(device, self.field)

        anariSetParameter(device, self.volume, 'value', ANARI_SPATIAL_FIELD, self.field)
        anariSetParameter(device, self.volume, 'valueRange', ANARI_FLOAT32_BOX1, [vmin, vmax])
        anariCommitParameters(device, self.volume)

        tf_size = 256
        rgb = self.tf_to_rgb(tf, tf_size, 1)
        alpha = self.tf_to_alpha(tf, tf_size, 1)
        color = np.zeros(tf_size*3,dtype=np.float32)
        opacity = np.zeros(tf_size,dtype=np.float32)
        for i in range(tf_size):
            color[i*3]   = rgb[i][0][0]
            color[i*3+1] = rgb[i][0][1]
            color[i*3+2] = rgb[i][0][2]
            opacity[i]   = alpha[i][0][0]
        self.set_array(device, self.volume, 'color', ANARI_FLOAT32_VEC3, tf_size, color)
        self.set_array(device, self.volume, 'opacity', ANARI_FLOAT32, tf_size, opacity)

        volumes = ffi.new('ANARIVolume[]', [self.volume])
        array = anariNewArray1D(device, volumes, ANARI_VOLUME, 1)
        anariSetParameter(device, self.world, 'volume', ANARI_ARRAY1D, array)

        anariCommitParameters(device, self.world)

