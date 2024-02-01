import os
import glob

import pickle

class WaymaxDataset():
    def __init__(
            self,
            data_path: str,
        ) -> None:
            """
            Initialize the Dataset object.

            Args:
                data_path (str): Path to the data directory.
                anchor_path (str): Path to the anchor file.
                max_object (int, optional): Maximum number of objects. Defaults to 32.
                max_map_points (int, optional): Maximum number of map points. Defaults to 3000.
                max_polylines (int, optional): Maximum number of polylines. Defaults to 256.
                history_length (int, optional): Length of history. Defaults to 11.
                num_points_polyline (int, optional): Number of points in each polyline. Defaults to 30.
            """
            
            super().__init__()
            
            self.data_path = data_path
            self.scenario_list = sorted(glob.glob(os.path.join(data_path, 'scenario*')))
            print("Total number of scenarios: ", len(self))
            
    def __len__(self) -> int:
        return len(self.scenario_list)
    
    def load_scenario_by_id(self, scenario_id: str):   
        """
        Load a scenario from the dataset by scenario id.

        Args:
            scenario_id (str): The scenario_id of the scenario to load.

        Returns:
            Tuple[str, Any]: A tuple containing the scenario ID and the loaded scenario.
        """
        # find index in file list
        filename = os.path.join(self.data_path, f'scenario_{scenario_id}.pkl')
        idx = self.scenario_list.index(filename)
        print("Loading scenario idx ", idx)
                
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
            scenario = cache['scenario']
        return scenario_id, scenario
    
    def load_scenario(self, index: int):   
        """
        Load a scenario from the dataset.

        Args:
            index (int): The index of the scenario to load.

        Returns:
            Tuple[str, Any]: A tuple containing the scenario ID and the loaded scenario.
        """
        filename = self.scenario_list[index]
        scenario_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
        
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
            scenario = cache['scenario']
        return scenario_id, scenario
        
    

        
    
        
        
        
            
        
    
    
        