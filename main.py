# import python modules
import os
import time
import yaml

# import isaacgym modules
from isaacgym import gymapi, gymutil

# import 3rd party modules
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from os import walk
class DataGenEnv(object):
    def __init__(self):
        # Initialize gym
        self.gym = gymapi.acquire_gym()
        
        # Parse agruments / isaacgym configuration
        self.args = gymutil.parse_arguments(
            description = 'Depth image dataset generation for 6D pose estimation',
            headless=True,
            no_graphics=True,
            custom_parameters=[
                {
                    "name": "--config",
                    "type": str,
                    "default": "cfg/config.yaml",
                    "help": " configuration file"
                },
                {
                    "name": "--save_results",
                    "action": "store_true",
                    "help": "save results to disk"
                }
            ],
        )
        
        config_file = self.args.config
        self.headless = self.args.headless
        self.save_results = self.args.save_results
        
        # Load configuration
        with open(config_file,"r") as f:
            cfg = yaml.safe_load(f)
        
        sim_cfg = cfg["simulation"]
        cam_cfg = sim_cfg["camera"]["AzureKinect"]['depth']
        
        # simulation setup (default)
        self.FILE_ZERO_PADDING_NUM = sim_cfg["FILE_ZERO_PADDING_NUM"]
        self.physics_engine = sim_cfg["physics_engine"]
        self.num_threads = sim_cfg["num_threads"]
        self.compute_device_id = sim_cfg["compute_device_id"]
        self.graphics_device_id = sim_cfg["graphics_device_id"]
        self.num_envs = sim_cfg['num_envs']
        self.use_gpu = sim_cfg['use_gpu']
        self.dt = sim_cfg["dt"]
        self.render_freq = 1/sim_cfg["render_freq"]
        self.num_iters = sim_cfg["num_iters"]
        self.target_object_name = sim_cfg["target_object"]
        self.target_dataset_name = sim_cfg["target_dataset"]
        self.object_rand_pose_range = sim_cfg["object_rand_pose_range"]
        self.camera_rand_position_range = sim_cfg["camera_rand_position_range"]
        self.camera_rand_rotation_range = sim_cfg["camera_rand_rotation_range"]
        
        
        self.target_object_name = sim_cfg["target_object"]
        self.target_dataset_name = sim_cfg["target_dataset"]
        self.base_object_name = sim_cfg["base_object"]
        self.base_dataset_name = sim_cfg["base_dataset"]
        
        
        # simulation method
        self.min_stable_pose_prob = sim_cfg['min_stable_pose_prob']
        self.max_num_stable_pose = sim_cfg['max_num_stable_pose']
        
        fx, fy, cx, cy = cam_cfg["fx"], cam_cfg["fy"], cam_cfg["cx"], cam_cfg["cy"]
        self.camera_matrix = np.array([[fx,0,cx],
                                       [0,fy,cy],
                                       [0, 0, 1]])
        
        # configure save directory
        root_dir = os.path.dirname(os.path.abspath(__file__)) + '/src'
        self.save_dir = os.path.join(root_dir,self.target_object_name)
        self.file_list = []
        
        for (dirpath, dirnames, filenames) in walk(self.save_dir):
            self.file_list.extend(filenames)
            break
        
        # save config files
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir,'config_yaml'),'w') as f:
            yaml.dump(cfg,f)
            
            
        # custom env variables
        self.envs = []
        self.hand_handles = []
        self.camera_handles = []
        self.object_handles = []
        self.base_handles = []
        self.default_dof_pos = []
        
        self.enable_viewer_sync = False
        self.task_status = None
        self.wait_timer = 0.0
        self.render_timer = self.render_freq + self.dt
        
        # Initialize sim
        self._create_sim()
        self._create_ground()
        self._create_viewer()
        self._create_envs()
        self.gym.prepare_sim(self.sim) # Prepares simulation with buffer allocations
        
        
    def _create_sim(self):
        # configure sim
        # check 'isaacgym.gymapi.SimParams' for more information
        sim_params = gymapi.SimParams()
        sim_params.dt = self.dt
        
        self.physics_engine = gymapi.SIM_PHYSX
        sim_params.substeps = 5
        sim_params.physx.solver_type = 4
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.num_threads
        sim_params.physx.use_gpu = self.use_gpu
        sim_params.physx.rest_offset = 0.0
        sim_params.use_gpu_pipeline = False
        # Set GPU pipeline
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # set up axis as Z-up
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # create sim
        self.sim = self.gym.create_sim( self.compute_device_id, self.graphics_device_id, self.physics_engine, sim_params)
        
    def _create_ground(self):
        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.distance = 0
        plane_params.static_friction = 0.3
        plane_params.dynamic_friction = 0.15
        plane_params.restitution = 0
        
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_viewer(self):
        if self.headless:
            self.viewer = None
        
        else:
            # create viewer
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            
            # set viewer camera pose
            self.cam_pose = gymapi.Vec3(0.5, 0.5, 2)
            cam_target = gymapi.Vec3(0.5, 0.5, 0)
            
            self.gym.viewer_camera_look_at(self.viewer,None,self.cam_pose, cam_target)
            
            # key callback
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_ESCAPE,"QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer,gymapi.KEY_V,"toggle_viewer_sync")
            
            if self.viewer is None:
                print("*** Failed to create viewer")
                quit()
            
    def __del__(self):
        self.gym.destroy_sim(self.sim)
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
            
    def _create_envs(self):
        asset_root = "./assets"
        
        #####################
        # load object asset #
        #####################
        
        object_asset_file = "{}/{}/{}.urdf".format(
            self.target_dataset_name,
            self.target_object_name,
            self.target_object_name
        )
        base_asset_file = "{}/{}/{}.urdf".format(
            self.base_dataset_name,
            self.base_object_name,
            self.base_object_name
        )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = False
        asset_options.thickness = 0.001
        asset_options.override_inertia = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        asset_options.vhacd_params.max_convex_hulls = 50
        asset_options.vhacd_params.max_num_vertices_per_ch = 1000
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)
        base_asset = self.gym.load_asset(self.sim, asset_root, base_asset_file, asset_options)
        
        # load object stable poses
        object_stable_prob = np.load("assets/{}/{}/stable_prob.npy".format( self.target_dataset_name, self.target_object_name))
        object_stable_poses = np.load("assets/{}/{}/stable_poses.npy".format( self.target_dataset_name, self.target_object_name))
        
        object_stable_prob = object_stable_prob[object_stable_prob > self.min_stable_pose_prob][:self.max_num_stable_pose]
        self.object_stable_prob = object_stable_prob / np.sum(object_stable_prob)
        object_stable_poses = object_stable_poses[:len(object_stable_prob)]
            
        self.object_stable_poses = []
        
        for pose in object_stable_poses:
            # 4x4 transform matrix to gymapi.Transform
            t = gymapi.Transform()
            t.p = gymapi.Vec3(pose[0,3], pose[1,3], pose[2,3])
            
            # 3x3 rotation matrix to quaternion
            r = R.from_matrix(pose[:3,:3])
            quat = r.as_quat()
            t.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])\
            
            self.object_stable_poses.append(t)
            
        ##############
        # Create Env #
        ##############
        
        # Configure env grid
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        env_lower = gymapi.Vec3(-0.1,-0.1,0.0)
        env_upper = gymapi.Vec3(0.1,0.1,0.002)
        
        print(f"Creating {self.num_envs} environments")
        self.cur_object_stable_poses = []
        for env_idx in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            # Add base
            base_pose = gymapi.Transform()
            base_pose.p = gymapi.Vec3(0, 0, 0)
            base_pose.r = gymapi.Quat(0,0,0,1)
            
            base_handle = self.gym.create_actor(env,base_asset,base_pose,"base", env_idx,1)
            
            # Add object
            object_pose = self.object_stable_poses[np.random.randint(len(self.object_stable_poses))]
            object_handle = self.gym.create_actor(env,object_asset,object_pose,"object", env_idx,1)
            
            # set segment id
            self.gym.set_rigid_body_segmentation_id(env,base_handle,0,0)
            self.gym.set_rigid_body_segmentation_id(env,object_handle,0,1)
            
            # add camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = int(self.camera_matrix[0,2] * 2.0)
            camera_props.height = int(self.camera_matrix[1,2] * 2.0)
            camera_props.horizontal_fov = 2*np.arctan2(self.camera_matrix[0,2], self.camera_matrix[0,0]) * 180/np.pi
            camera_props.far_plane = 1
            camera_handle  = self.gym.create_camera_sensor(env,camera_props)
            
            # gym camera pose def: x = optical axis, y = left, z = down | convention: OpenGL
            self.cam_pose = gymapi.Transform()
            self.cam_pose.p = gymapi.Vec3(0,0.1,0.3)
            self.cam_pose.r = gymapi.Quat.from_euler_zyx(0,np.deg2rad(90 + 13.5),np.pi/2)
            self.gym.set_camera_transform(camera_handle,env,self.cam_pose)
            
            # get camera extrinsic scene
            if env_idx == 0:
                # convert z = x, x = -y, y = -z
                rot = R.from_quat([self.cam_pose.r.x, self.cam_pose.r.y, self.cam_pose.r.z, self.cam_pose.r.w]).as_matrix()
                rot_convert = np.array([[0,0,1], [-1,0,0], [0,-1,0]])
                rot = np.dot(rot,rot_convert)
                self.camera_extr = np.eye(4)
                self.camera_extr[:3,:3] = rot
                self.camera_extr[:3,3] = np.array([self.cam_pose.p.x, self.cam_pose.p.y, self.cam_pose.p.z])
                
            # append handles
            self.envs.append(env)
            self.cur_object_stable_poses.append(object_pose)
            self.object_handles.append(object_handle)
            self.camera_handles.append(camera_handle)
            self.base_handles.append(base_handle)
            
            
      
    def reset_env(self):
        # reset objects to random stable poses
        reset_object_poses = []
        reset_camera_poses = []
        for env_idx in range(self.num_envs):
            ##########################################
            # Reset object pose to random stable poses
            ##########################################
            rigid_body_handle = self.gym.get_actor_rigid_body_handle(self.envs[env_idx],self.object_handles[env_idx], 0)
            # get stable pose
            t_stable = self.cur_object_stable_poses[env_idx]
            p_stable = t_stable.p
            q_stable = t_stable.r

            # get random pose
            p_random = gymapi.Vec3(
                np.random.uniform(-self.object_rand_pose_range, self.object_rand_pose_range),
                np.random.uniform(-self.object_rand_pose_range, self.object_rand_pose_range),
                0.021,)
            
            # Give constraints of random rotation for symmetric stable poses
            q_random = R.from_euler('z', np.random.uniform(0, 2*np.pi), degrees=False).as_quat()
            q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])

            # get random stable pose
            p = q_random.rotate(p_stable) + p_random
            q = (q_random*q_stable).normalize()
            t = gymapi.Transform(p, q)
            
            self.gym.set_rigid_transform(self.envs[env_idx], rigid_body_handle, t)
            
            # append pose
            reset_object_poses.append(t)

            ####################################
            # Reset camera pose to random  poses
            ####################################
            # get stable pose
            p_stable = self.cam_pose.p
            q_stable = self.cam_pose.r

            # get random pose
            p_random = gymapi.Vec3(
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range),
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range),
                np.random.uniform(-self.camera_rand_position_range, self.camera_rand_position_range),)
            
            # Give constraints of random rotation for symmetric stable poses
            q_random = R.from_euler('z', np.random.uniform(-self.camera_rand_rotation_range, self.camera_rand_rotation_range)
                                    , degrees=True).as_quat()
            q_random = gymapi.Quat(q_random[0], q_random[1], q_random[2], q_random[3])

            # get random stable pose
            p = q_random.rotate(p_stable) + p_random
            q = (q_random*q_stable).normalize()
            cam_pose = gymapi.Transform(p, q)
            self.gym.set_camera_transform(self.camera_handles[env_idx],self.envs[env_idx],cam_pose)
            
            rot = R.from_quat([cam_pose.r.x, cam_pose.r.y, cam_pose.r.z, cam_pose.r.w]).as_matrix()
            rot_convert = np.array([[0,0,1], [-1,0,0], [0,-1,0]])
            rot = np.dot(rot,rot_convert)
            camera_extr = np.eye(4)
            camera_extr[:3,:3] = rot
            camera_extr[:3,3] = np.array([cam_pose.p.x, cam_pose.p.y, cam_pose.p.z])
            
            reset_camera_poses.append(camera_extr)
            
            ######################################
            # Reset object color to random  colors
            ######################################
            color_object = gymapi.Vec3( np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1))
            color_base = gymapi.Vec3( np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1))
            self.gym.set_rigid_body_color(self.envs[env_idx], self.object_handles[env_idx], 0, gymapi.MESH_VISUAL_AND_COLLISION,color_object)
            self.gym.set_rigid_body_color(self.envs[env_idx], self.base_handles[env_idx], 0, gymapi.MESH_VISUAL_AND_COLLISION,color_base)
            # # disable gravity
            # obj_props = self.gym.get_actor_rigid_body_properties(self.envs[env_idx], self.camera_handles[env_idx])
            # obj_props[0].flags = gymapi.RIGID_BODY_DISABLE_GRAVITY
            # self.gym.set_actor_rigid_body_properties(self.envs[env_idx], self.camera_handles[env_idx], obj_props, False)
            
        # delete all existing lines
        if self.viewer:
            self.gym.clear_lines(self.viewer)

        return reset_object_poses, reset_camera_poses
    
    def plot(self,imgs_or_masks):
        fig = plt.figure(figsize=(8,8))
        columns = 10
        rows = 10
        for i in range(1,self.num_envs + 1):
            fig.add_subplot(columns,rows, i)
            plt.imshow(imgs_or_masks[i-1])
        plt.show()
        
    def depth_to_pcd(self, depth_img):
            height, width = depth_img.shape
            row_indices = np.arange(height)
            col_indices = np.arange(width)
            pixel_grid = np.meshgrid(col_indices, row_indices)
            pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
            pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
            depth_arr = np.tile(depth_img.flatten(), [3, 1])
            point_cloud = depth_arr * np.linalg.inv(self.camera_matrix).dot(pixels_homog)
            
            point_cloud = point_cloud
            return point_cloud.transpose()
        
    def step(self, n_step = 0):
        # reset env
        object_poses, cam_poses = self.reset_env()
        object_poses = self.pose_type_conversion(object_poses)
        # step the physics
        self.gym.simulate(self.sim)
        # refresh results
        self.gym.fetch_results(self.sim, True)
        # step rendering
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)

        # get depth image
        depth_images, color_images, segmasks = self.get_camera_image(n_step)
        
        # depth_images[10] = np.multiply(depth_images[10], segmasks[10])
        self.world2camera_frame = np.linalg.inv(self.camera_extr)
        
        # save data
        if self.save_results:
            
            os.makedirs(os.path.join(self.save_dir,'data'), exist_ok=True)
            
            for env_idx in range(self.num_envs):
                name = ("_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.npy')%(n_step * self.num_envs + env_idx)
                # name_img = ("_%0" + str(self.FILE_ZERO_PADDING_NUM) + 'd.png')%(n_step * self.num_envs + env_idx)
                with open(os.path.join(self.save_dir,'data','depth_image' + name), 'wb') as f:
                    np.save(f, depth_images[env_idx])
                with open(os.path.join(self.save_dir,'data','color_image' + name), 'wb') as f:
                    np.save(f, color_images[env_idx])
                with open(os.path.join(self.save_dir,'data','mask' + name), 'wb') as f:
                    np.save(f, segmasks[env_idx])
                with open(os.path.join(self.save_dir,'data','pose' + name), 'wb') as f:
                    np.save(f, self.pose_camera_frame(object_poses[env_idx], cam_poses[env_idx]))
                    
                
        return None
    
    def pose_type_conversion(self,object_poses):
        converted_poses = []
        for object_pose in object_poses:
            p = object_pose.p
            q = object_pose.r
            converted_pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w],dtype=np.float32)
            converted_poses.append(converted_pose)
        return converted_poses
    
    def pose_camera_frame(self, pose, cam_pose):
        
        p = pose[:3]
        q = pose[3:]
        
        r = R.from_quat(q).as_matrix()
        object2world_frame = np.vstack([np.c_[r, p.reshape(3,1)], np.array([[0,0,0,1]])])
        obj_pose_hom = np.linalg.inv(cam_pose) @ object2world_frame
        
        r = obj_pose_hom[:3,:3]
        t = obj_pose_hom[:3, 3].reshape(3)
        q = R.from_matrix(r).as_quat()
        q = q / np.linalg.norm(q)
        new_pose = np.r_[t, q]
        
        return new_pose
    
    def visualize_camera_axis(self):
        """Visualize camera axis"""
        # draw camera pose with line
        camera_pose = self.camera_extr

        # 4x4 matrix to gymapi.Transform
        rot = camera_pose[:3, :3]
        quat = R.from_matrix(rot).as_quat()
        pose = gymapi.Transform()
        pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])
        pose.p = gymapi.Vec3(camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3])

        # draw line
        axes_geom = gymutil.AxesGeometry(100)
        for env_idx in range(self.num_envs):
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[env_idx], pose)
        

    def get_camera_image(self, n_step):
        """Get images from camera

        Returns:
            depth_images (numpy.ndarray): image of shape (num_envs, H, W, 3)
            segmasks (numpy.ndarray): segmentation mask of shape (num_envs, H, W)
        """
        depth_images = []
        color_images = []
        segmasks = []
        self.gym.render_all_camera_sensors(self.sim)
        for i in range(self.num_envs):
            depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
            color_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            segmask     = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_SEGMENTATION)
            # Change data type for lighter storage
            depth_image = np.array(depth_image, dtype = np.float32)
            color_image = np.array(color_image, dtype = np.float32)
            segmask     = np.array(segmask, dtype = np.bool8)
            
            image = np.zeros((*depth_image.shape,3))
            # RGB
            for i in range(int(color_image.shape[1]/4)):
                image[:,i,0] = color_image[:,i*4] 
                image[:,i,1] = color_image[:,i*4+1]
                image[:,i,2] = color_image[:,i*4+2]
                # image[:,i,3] = color_image[:,i*4+3]
            color_image = image.astype(np.uint8)
            depth_images.append(depth_image)
            color_images.append(color_image)
            
            segmasks.append(segmask)
            
        depth_images = np.array(depth_images) * -1
        color_images = np.array(color_images)
        segmasks = np.array(segmasks)
        
        return depth_images, color_images, segmasks

    

    
if __name__ == '__main__':
    env = DataGenEnv()
    config_file = os.path.join(os.path.dirname(__file__), 'cfg/config.yaml')
    with open(config_file,"r") as f:
        cfg = yaml.safe_load(f)
        
    sim_cfg = cfg["simulation"]
    num_envs = sim_cfg['num_envs']
    for i in range(env.num_iters):
        start_time = time.time()
        # env.visualize_camera_axis()
        env.step(i)
        print('step: {} | Num of data: {}  | time: {:.3f}'.format(i, i*num_envs, time.time() - start_time))
