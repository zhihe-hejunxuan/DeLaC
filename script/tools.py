import numpy as np
import torch
from pymatgen.core import Lattice, Structure, Element
# 统一到 numpy
def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def build_structures_from_batch(
    batch_data,
    angles_in_radians=False,   # 你的管线里角度大概率是弧度；若已是度数，改成 False
    zero_based_types=False    # 若元素类型是 0-based（0..99），设 True，会自动 +1 变成原子序数
):
    num_atoms, atom_types, lengths, angles, frac_coords = batch_data.num_atoms, batch_data.atom_types, batch_data.lengths, batch_data.angles, batch_data.frac_coords
    return build_structures_for_m3gnet(
        num_atoms=num_atoms,
        atom_types=atom_types,
        lengths=lengths,
        angles=angles,
        frac_coords=frac_coords,
        angles_in_radians=angles_in_radians,
        zero_based_types=zero_based_types
    )

def build_structures_for_m3gnet(
    num_atoms,
    atom_types,
    lengths,
    angles,
    frac_coords,
    *,
    angles_in_radians=False,   # 你的管线里角度大概率是弧度；若已是度数，改成 False
    zero_based_types=False    # 若元素类型是 0-based（0..99），设 True，会自动 +1 变成原子序数
):
    """
    将批量晶体（拼接在一起的张量）转换为 pymatgen.Structure 列表。
    适配 torch/numpy 两种输入；自动把角度从弧度→度（可选）；把分数坐标规整到单胞内。

    参数形状约定：
      - num_atoms:       [B]
      - atom_types:      [sum(num_atoms)]，每个是原子序数Z或0-based类型
      - lengths:         [B, 3]  (a, b, c)
      - angles:          [B, 3]  (alpha, beta, gamma)  默认认为是弧度
      - frac_coords:     [sum(num_atoms), 3]  (分数坐标)
    """

    num_atoms   = to_np(num_atoms).astype(int)
    atom_types  = to_np(atom_types).astype(int)
    lengths     = to_np(lengths).astype(float)
    angles      = to_np(angles).astype(float)
    frac_coords = to_np(frac_coords).astype(float)

    if zero_based_types:
        atom_types = atom_types + 1  # 转为真实的原子序数 Z

    if angles_in_radians:
        angles = np.degrees(angles)  # pymatgen 需要角度(度)

    # 构建结构
    structures = []
    start = 0
    for i, n in enumerate(num_atoms.tolist()):
        n = int(n)
        z_slice   = atom_types[start:start + n]
        coords    = frac_coords[start:start + n]
        a_b_c     = lengths[i].tolist()
        alpha_b_g = angles[i].tolist()

        # 规整分数坐标到单胞内（避免越界）
        # coords = coords - np.floor(coords)

        # 构建晶格和物种
        lattice = Lattice.from_parameters(
            a_b_c[0],
            a_b_c[1],
            a_b_c[2],
            alpha_b_g[0],
            alpha_b_g[1],
            alpha_b_g[2],
        )
        species  = [Element.from_Z(int(z)) for z in z_slice]
        s = Structure(lattice=lattice, species=species, coords=coords, coords_are_cartesian=False, to_unit_cell=True)
        structures.append(s)

        start += n

    return structures
