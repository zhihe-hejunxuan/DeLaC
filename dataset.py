import torch, os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pymatgen.core import Structure
from utils import build_crystal_graph, build_crystal
from pymatgen.core.periodic_table import Element

class MatDataset(Dataset):
    def __init__(self, data_path, prop_name=None, graph_method='crystalnn', pt_path=None):
        super().__init__()
        
        self.graph_method = graph_method
        self.prop_name = prop_name

        # if data_path[-3:] == '.pt':
        #     self.data_dict_list = torch.load(data_path, weights_only=False)
        if data_path[-3:] == '.pt':
            self.data_dict_list = torch.load(data_path, weights_only=False)
            # 过滤
            old_len = len(self.data_dict_list)
            self.data_dict_list = [d for d in self.data_dict_list if d["graph_arrays"][-1] <= 20]
            new_len = len(self.data_dict_list)
            if new_len < old_len:
                print(f"[INFO] 过滤 num_atoms > 20 的样本: 从 {old_len} 条 -> {new_len} 条")
                # Filter in memory; do not overwrite the input cache.
        elif data_path[-4:] == '.csv':
            self.df = pd.read_csv(data_path)
            self._preprocess(data_path[:-4] + '.pt')
        elif os.path.isdir(data_path):
            self._process_directory(data_path, pt_path)
        else:
            raise NotImplementedError
        
    def _process_directory(self, dir_path, pt_path):
        data_dict_list = []
        cif_files = [f for f in os.listdir(dir_path) if f.endswith('.cif')]

        for cif_file in tqdm(cif_files, desc="Processing CIF files"):
            try:
                cif_path = os.path.join(dir_path, cif_file)
                with open(cif_path, 'r') as file:
                    crystal_str = file.read()
                crystal = build_crystal(crystal_str, niggli=True, primitive=False)
                graph_arrays = build_crystal_graph(crystal, graph_method=self.graph_method)

                data_dict = {
                    'mp_id': -100,
                    'formation_energy_per_atom': -100.0,
                    'cif': crystal_str,
                    'graph_arrays': graph_arrays
                }

                data_dict_list.append(data_dict)
            except:
                pass
        
        self.data_dict_list = data_dict_list
        torch.save(self.data_dict_list, pt_path)

    # def _preprocess(self, preprocessed_file_path):
    #     data_dict_list = []
    #     if 'num_sites' in self.df.columns and 'energy_above_hull' in self.df.columns:
    #         old_length = len(self.df)
    #         self.df = self.df[(self.df['num_sites'] <= 20) & (self.df['energy_above_hull'] <= 0.1)]
    #         print(f"[INFO] 筛选 num_sites <= 20 且 energy_above_hull <= 0.1, 从 {old_length} 行到 {len(self.df)} 行")
    #     for i in tqdm(range(len(self.df))):
    #         try:
    #             data_dict = self._get_mat_data(i)
    #             data_dict_list.append(data_dict)
    #         except:
    #             pass
        
    #     self.data_dict_list = data_dict_list
    #     torch.save(self.data_dict_list, preprocessed_file_path)


    def _preprocess(self, preprocessed_file_path):
        # 定义禁止元素

        def has_forbidden_elements(structure) -> bool:
            """
            20260207 appended: 检查结构中是否含有禁止元素
            https://github.com/microsoft/mattergen/blob/main/data-release/mp-20/README.md
            """
            for el in structure.composition.elements:
                Z = Element(el.symbol).Z
                if el.symbol in {"Tc", "Pm", "Ne", "He"}:
                    return True
                if Z >= 84:
                    return True
            return False
        

        data_dict_list = []
        if 'num_sites' in self.df.columns and 'energy_above_hull' in self.df.columns:
            old_length = len(self.df)
            self.df = self.df[(self.df['num_sites'] <= 20) & (self.df['energy_above_hull'] <= 0.1)]
            print(f"[INFO] 筛选 num_sites <= 20 且 energy_above_hull <= 0.1, 从 {old_length} 行到 {len(self.df)} 行")

        old_length = len(self.df)
        # 进一步过滤掉含禁止元素的行
        keep_rows = []
        for i in range(len(self.df)):
            try:
                crystal_str = self.df.iloc[i]['cif']
                structure = Structure.from_str(crystal_str, fmt="cif")
                if has_forbidden_elements(structure):
                    continue
                keep_rows.append(i)
            except:
                # 如果解析出错直接跳过
                continue
        self.df = self.df.iloc[keep_rows].reset_index(drop=True)
        print(f"[INFO] 过滤掉含禁止元素的样本: 从 {old_length} 行到 {len(self.df)} 行")

        # 遍历构建图
        for i in tqdm(range(len(self.df))):
            try:
                data_dict = self._get_mat_data(i)
                data_dict_list.append(data_dict)
            except:
                pass

        self.data_dict_list = data_dict_list
        torch.save(self.data_dict_list, preprocessed_file_path)

        

    def _get_mat_data(self, index):
        if hasattr(self, 'data_dict_list'):
            return self.data_dict_list[index]

        row = self.df.iloc[index]
        crystal_str = row['cif']
        crystal = build_crystal(crystal_str, niggli=True, primitive=False)
        graph_arrays = build_crystal_graph(crystal, graph_method=self.graph_method)

        data_dict = {
            'mp_id': row['material_id'],
            'cif': crystal_str,
            'graph_arrays': graph_arrays
        }

        if self.prop_name in row.keys():
            data_dict[self.prop_name] = row[self.prop_name]
        
        if self.prop_name == "formation_energy_per_atom":
            if "formation_energy_per_atom" not in data_dict:
                data_dict['formation_energy_per_atom'] = -100.0
        
        return data_dict

    
    def __len__(self):
        if hasattr(self, 'data_dict_list'):
            return len(self.data_dict_list)
        return len(self.df)
    
    
    def __getitem__(self, index):
        data_dict = self._get_mat_data(index)
        prop = torch.tensor(data_dict[self.prop_name])
        (frac_coords, atom_types, scaled_lengths, lengths, angles, edge_indices, to_jimages, num_atoms) = data_dict['graph_arrays']
        data = Data(
            frac_coords=torch.Tensor(frac_coords).float(),
            atom_types=torch.LongTensor(atom_types),
            scaled_lengths=torch.Tensor(scaled_lengths).view(1, -1),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        return data
            