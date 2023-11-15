import numpy as np
import torch
import os
from typing import List, Dict, Callable, Optional
from mtr.config import cfg, cfg_from_yaml_file
from mtr.models import model as model_utils
from mtr.datasets.waymo.waymo_dataset import WaymoDataset
# from mtr.datasets.waymo.generate_graph import generate_map_graph
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from mtr.utils import common_utils

from .visualization.vis_utils import plot_map, plot_signal, plot_traj_with_time, plot_obj_pose, plot_traj_with_speed
from .mtr_lightning import MTR_Lightning, PrintLogger 

class MTRInference():
    def __init__(self, cfg_file: str) -> None:
        print("=========== MTR Inference ===========")
        self.cfg = cfg
        cfg_from_yaml_file(cfg_file, self.cfg)
    
        ### Build Dataset ###
        self.dataset = WaymoDataset(self.cfg.DATA_CONFIG, training=False, logger=None)
        
        ### Build Model ###
        # self.model = model_utils.MotionTransformer(config=cfg.MODEL)
        self.model = MTR_Lightning(cfg)
        
        
        ### Load Checkpoint ###
        # _ = self.model.load_params_from_file(filename=ckpt_path, to_cpu=False)
        
        # if torch.cuda.is_available():
        #     self.model = self.model.cuda()
        
        # self.model.eval()
    def load_from_checkpoint(self, ckpt_path: str):
        self.model = self.model.load_from_checkpoint(ckpt_path)
        
    def load_from_params(self, params_path: str):
        self.model.model.load_params_from_file(params_path)
        self.model.cuda()   
             
    def generate_info(self, index):
        return self.dataset.load_info(index)
    
    # def generate_graph(self, index: int):
    #     scene_id, info = self.generate_info(index)
    #     map_graph = generate_map_graph(info)
    #     return scene_id, map_graph
    
    def generate_input_data(self, index: int, shift: int = 0, preprocess: Optional[Callable] = None)-> Dict:
        '''
        This function generates the input data for the model.
        
        Args:
            index (int): The index of the data in the dataset.
            shift (int): The number of frames to shift the data by.
        Returns:
            data (dict): The input data for the model.
        '''
        # Load data from cache
        scene_id, info = self.generate_info(index)
        
        # Extract data from scene
        data = self.dataset.extract_scene_data(scene_id, info, shift)
        
        # Preprocess data
        if preprocess is not None:
            data = preprocess(data)
        
        # Make data in a batch with batch size 1
        data_batch = self.dataset.collate_batch([data])
        
        return scene_id, info, data_batch
        
    def inference(self, batch_dict: Dict) -> List[Dict]:
        '''
        This function runs inference on the model.
        
        Args:
            batch_dict (dict): The input data for the model.
        Returns:
            final_pred_dicts List(dict): The output of the model.
        '''
        self.model.eval()
        with torch.no_grad():
            batch_pred_dicts = self.model(batch_dict)
        print(batch_pred_dicts.keys())
        batch_pred_dicts = self.generate_prediction_dicts(batch_pred_dicts)
        return batch_pred_dicts
        
    def sample_control(self, batch_dict: Dict) -> List[Dict]:
        batch_pred_dicts = self.inference(batch_dict, generate_prediction=False)
    
    def generate_prediction_dicts(self, batch_dict, output_path=None):
        """
        Args:
            batch_dict:
                pred_scores: (num_center_objects, num_modes)
                pred_trajs: (num_center_objects, num_modes, num_timestamps, 7)

            input_dict:
                center_objects_world: (num_center_objects, 10)
                center_objects_type: (num_center_objects)
                center_objects_id: (num_center_objects)
                center_gt_trajs_src: (num_center_objects, num_timestamps, 10)
        """
        input_dict = batch_dict['input_dict']

        pred_trajs = batch_dict['pred_trajs']
        center_objects_world = input_dict['center_objects_world'].type_as(pred_trajs)

        num_center_objects, num_modes, num_timestamps, num_feat = pred_trajs.shape
        assert num_feat == 7
        
        pred_trajs_xy = pred_trajs[:, :, :, 0:2].view(num_center_objects, num_modes * num_timestamps, 2)
        pred_trajs_v = pred_trajs[:, :, :, 5:7].view(num_center_objects, num_modes * num_timestamps, 2)
                
        pred_trajs_xy = common_utils.rotate_points_along_z(
            points=pred_trajs_xy,
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, 2)
        pred_trajs_xy  = pred_trajs_xy+ center_objects_world[:, None, None, 0:2]
        
        pred_trajs_v = common_utils.rotate_points_along_z(
            points=pred_trajs_v,
            angle=center_objects_world[:, 6].view(num_center_objects)
        ).view(num_center_objects, num_modes, num_timestamps, 2)
        
        pred_trajs_world = torch.cat([pred_trajs_xy, pred_trajs_v], dim=-1)        

        batch_dict['pred_trajs_world'] = pred_trajs_world.detach().cpu().numpy()
        
        return batch_dict
    
    def plot_result(self, scene_id: str, info: dict, final_pred_dicts: dict, shift: int = 0, plot_gt: bool = False):
        # Visualize
        fig, ax = plot_map(info['map_infos'])

        t = info['current_time_index'] + shift
        
        dynamic_map_infos = info['dynamic_map_infos']
        plot_signal(dynamic_map_infos, t, ax)

        track_infos = info['track_infos']
        pred_trajs = final_pred_dicts['pred_trajs_world']
        pred_scores = final_pred_dicts['pred_scores'].cpu().numpy()
        for pred_traj, score in zip(pred_trajs, pred_scores):     
            for future, s in zip(pred_traj, score):
                ax.plot(future[:, 0], future[:, 1], color='xkcd:russet', linewidth=2, linestyle='-', alpha=s*0.7+0.3, zorder=2)
        
        for obj_idx in info['tracks_to_predict']['track_index']:
            plot_traj_with_speed([track_infos['object_type'][obj_idx]], [track_infos['trajs'][obj_idx][:t]], ax=ax, fig=fig,)
                
        if plot_gt:
            for obj_idx in info['tracks_to_predict']['track_index']:
                plot_traj_with_time([track_infos['object_type'][obj_idx]], [track_infos['trajs'][obj_idx]], info['timestamps_seconds'], ax=ax, fig=fig,)
                
        for obj_type, traj in zip(
            info['track_infos']['object_type'], info['track_infos']['trajs']
        ):
            plot_obj_pose(obj_type, traj[t-1], ax=ax)
            
        ax.set_title(f'Scene {scene_id} at {t/10} seconds')
        return fig, ax
        
    def visualize(self, scene_id: str, info: dict, shift: int = 0, plot_gt: bool = False):
        '''
        Get the input data, run inference, and visualize the results.
        '''
        data = self.dataset.extract_scene_data(scene_id, info, shift)
        
        # Make data in a batch with batch size 1
        data_batch = self.dataset.collate_batch([data])
        
        # Run inference
        final_pred_dicts = self.inference(data_batch)
        
        fig, ax = self.plot_result(scene_id, info, final_pred_dicts, shift, plot_gt)
        
        return scene_id, fig, ax
    
    def generate_gif(self, index: int, outut_dir: str = '.'):
        
        scene_id, info = self.dataset.load_info(index)
        
        # make temp dir
        os.makedirs('temp', exist_ok=True)
        file_list = []
        for shift in range(81):
            try:
                scene_id, fig, ax = self.visualize(scene_id, info, shift=shift)
                
                if shift == 0:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                else:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    
                filename = f'temp/{shift}.png'
                fig.savefig(filename)
                file_list.append(filename)
                plt.close(fig)
            except Exception as e:
                if e is KeyboardInterrupt:
                    raise e
                else:
                    break
                
        images = []
        
        for file in file_list:
            images.append(imageio.imread(file))
        
        # check if output dir exists
        if not os.path.isdir(outut_dir):
            
            os.makedirs(outut_dir, exist_ok=True)
            
        output_file = os.path.join(outut_dir, f'{index:05}_{scene_id}.gif')
        imageio.mimsave(output_file, images, duration = 0.5)
            
        for root, dirs, files in os.walk('temp', topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
                
                
if __name__ == '__main__':
    cfg_file = '/home/zixuzhang/Documents/Git/MTR/tools/cfgs/waymo/eval.yaml'
    ckpt_path = '/home/zixuzhang/Documents/Git/MTR/model/checkpoint_epoch_30.pth'
    
    mtr_inference = MTRInference(cfg_file, ckpt_path)
    
    for i in tqdm(range(len(mtr_inference.dataset))):
        try:
            mtr_inference.generate_gif(i, 'gif_interactive')
        except Exception as e:
            if e == KeyboardInterrupt:
                raise e
            else:
                print(f'Error on {i}, {e}')
            
        