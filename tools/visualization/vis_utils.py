import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

mpl.rcParams['hatch.linewidth'] = 4  # previous svg hatch linewidth
from matplotlib.patches import Polygon, RegularPolygon, Circle
from matplotlib.collections import LineCollection
import numpy as np
from typing import Dict, List, Tuple
from .vis_config_bright_hd import canvas_config, road_line_config, road_edge_config, speed_bump_config, \
    crosswalk_config, lane_config, stop_sign_config, object_config, signal_config, driveway_config
from core.utils import PyLaneletMap

v_max = 10
v_min = 0


def setup_canvas():
    fig = plt.figure(figsize=(canvas_config['width'], canvas_config['width']))
    ax = fig.add_subplot(111)
    ax.set_facecolor(canvas_config['background_color'])
    ax.set_aspect('equal')
    if not canvas_config['tick_on']:
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labeltop=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()#pad=0)
    return fig, ax


def _plot_line(
    points: np.ndarray, config: Dict, ax: plt.Axes = None, color: str = None, linewidth: float = None,
    linestyle: str = None, alpha: float = None
):
    if ax is None:
        ax = plt.gca()
    # override config
    color = config['color'] if color is None else color
    linewidth = config['linewidth'] if linewidth is None else linewidth
    linestyle = config['linestyle'] if linestyle is None else linestyle
    alpha = config['alpha'] if alpha is None else alpha

    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=2,
    )


def _plot_broken_line(
    points: np.ndarray,
    config: Dict,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override config
    color = config['color'] if color is None else color
    linewidth = config['linewidth'] if linewidth is None else linewidth
    linestyle = config['linestyle'] if linestyle is None else linestyle
    alpha = config['alpha'] if alpha is None else alpha

    n_broken = 8
    skip = 2
    n_points = int(points.shape[0] / n_broken) * n_broken
    point_x = points[:n_points, 0].reshape(-1, n_broken).T
    point_y = points[:n_points, 1].reshape(-1, n_broken).T

    ax.plot(
        point_x[:-skip, :],
        point_y[:-skip, :],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )


def plot_road_line(
    road_line: Dict,
    polylines: np.ndarray,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    line_type = road_line['type']
    config = road_line_config[line_type]
    
    polylines_start, polylines_end = road_line['polyline_index']
    points = polylines[polylines_start:polylines_end, :2]
    
    if 'BROKEN' in line_type:
        _plot_broken_line(
            points=points,
            config=config,
            ax=ax,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
    else:
        _plot_line(
            points=points,
            config=config,
            ax=ax,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )


def plot_road_edge(
    road_edge: Dict,
    polylines: np.ndarray,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    config = road_edge_config[road_edge['type']]
    polylines_start, polylines_end = road_edge['polyline_index']
    points = polylines[polylines_start:polylines_end, :2]

    _plot_line(
        points=points,
        config=config,
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )


def plot_speed_bump(
    speed_bump: Dict,
    polylines: np.ndarray,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
):
    return None

    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = speed_bump_config['facecolor'] if facecolor is None else facecolor
    edgecolor = speed_bump_config['edgecolor'] if edgecolor is None else edgecolor
    alpha = speed_bump_config['alpha'] if alpha is None else alpha

    polylines_start, polylines_end = speed_bump['polyline_index']
    points = polylines[polylines_start:polylines_end, :2]
    
    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0,
        alpha=alpha,
        hatch=r'//',
        zorder=2,
    )

    ax.add_patch(p)


def plot_crosswalk(
    crosswalk: Dict,
    polylines: np.ndarray,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = crosswalk_config['facecolor'] if facecolor is None else facecolor
    edgecolor = crosswalk_config['edgecolor'] if edgecolor is None else edgecolor
    alpha = crosswalk_config['alpha'] if alpha is None else alpha

    polylines_start, polylines_end = crosswalk['polyline_index']
    points = polylines[polylines_start:polylines_end, :2]

    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2,
        alpha=alpha,
        hatch=r'//',
        zorder=2,
    )

    ax.add_patch(p)


def plot_stop_sign(
    point: np.ndarray,
    ax: plt.Axes = None,
    radius: float = None,
    facecolor: str = None,
    edgecolor: str = None,
    linewidth: float = None,
    alpha: float = None,
):
    if ax is None:
        ax = plt.gca()
    # override default config
    facecolor = stop_sign_config['facecolor'] if facecolor is None else facecolor
    edgecolor = stop_sign_config['edgecolor'] if edgecolor is None else edgecolor
    linewidth = stop_sign_config['linewidth'] if linewidth is None else linewidth
    radius = stop_sign_config['radius'] if radius is None else radius
    alpha = stop_sign_config['alpha'] if alpha is None else alpha
    
    point = point.reshape(-1)

    p = RegularPolygon(
        point,
        numVertices=6,
        radius=radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )

    ax.add_patch(p)


def plot_lane(
    lane: Dict,
    polylines: np.ndarray,
    ax: plt.Axes = None,
    color: str = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    config = lane_config[lane['type']]
    
    polylines_start, polylines_end = lane['polyline_index']
    points = polylines[polylines_start:polylines_end, :2] 
    
    _plot_line(
        points=points,
        config=config,
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
    )


def plot_driveway(
    driveway: Dict,
    ax: plt.Axes = None,
    facecolor: str = None,
    edgecolor: str = None,
    linewidth: float = None,
    alpha: float = None,
):
    points = driveway['polyline'][:, :2]

    facecolor = driveway_config['facecolor'] if facecolor is None else facecolor
    edgecolor = driveway_config['edgecolor'] if edgecolor is None else edgecolor
    linewidth = driveway_config['linewidth'] if linewidth is None else linewidth
    alpha = driveway_config['alpha'] if alpha is None else alpha

    p = Polygon(
        points,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(p)


def plot_traj_with_speed(
    obj_type_list: list,
    trajs: np.ndarray,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
    show_colorbar: bool = False,
):
    '''
    This function plot trajectory with speed as color gradient
    '''
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # plot color line
    norm = plt.Normalize(v_min, v_max)

    # traj have feature [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    for obj_type, traj in zip(obj_type_list, trajs):
        config = object_config[obj_type]
        val = traj[:, -1] == 1
        points = traj[val, :2].reshape(-1, 1, 2)  # (N, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
        speed = np.linalg.norm(traj[val, -3:-1], axis=1)

        # override config
        linewidth = config['linewidth'] if linewidth is None else linewidth
        linestyle = config['linestyle'] if linestyle is None else linestyle
        alpha = config['alpha'] if alpha is None else alpha

        lc = LineCollection(segments, cmap='viridis', norm=norm, linestyle=linestyle, alpha=alpha, zorder=3)
        # Set the values used for colormapping
        lc.set_array(speed)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
    if show_colorbar:
        fig.colorbar(line, ax=ax, label='speed (m/s)', location='right', shrink=0.3, pad=0.02)


def plot_traj_with_time(
    obj_type_list: list,
    trajs: np.ndarray,
    timestamps_seconds: list,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    linewidth: float = None,
    linestyle: str = None,
    alpha: float = None,
):
    '''
    This function plot trajectory with time as color gradient
    '''
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    # plot color line
    norm = plt.Normalize(timestamps_seconds[0], timestamps_seconds[-1])

    # traj have feature [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    for obj_type, traj in zip(obj_type_list, trajs):
        config = object_config[obj_type]
        val = traj[:, -1] == 1
        points = traj[val, :2].reshape(-1, 1, 2)  # (N, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1, 2, 2)
        valid_time = np.array(timestamps_seconds)[val]

        # override config
        linewidth = config['linewidth'] if linewidth is None else linewidth
        linestyle = config['linestyle'] if linestyle is None else linestyle
        alpha = config['alpha'] if alpha is None else alpha

        lc = LineCollection(segments, cmap='viridis', norm=norm, linestyle=linestyle, alpha=alpha, zorder=3)
        # Set the values used for colormapping
        lc.set_array(valid_time)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)

    # fig.colorbar(line, ax=ax, label='TImeStamp (s)', location='right', shrink=0.3, pad=0.02)


def plot_obj_pose(
    obj_type: str,
    state: np.ndarray,
    ax: plt.Axes = None,
    facecolor: float = None,
    alpha: float = None,
):
    # state have feature [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    if state[-1] == 0:
        # Ignore invalid object
        return
    config = object_config[obj_type]
    if ax is None:
        ax = plt.gca()

    facecolor = config['facecolor'] if facecolor is None else facecolor
    alpha = config['alpha'] if alpha is None else alpha

    # if obj_type == 'TYPE_PEDESTRIAN':
    #     p = Circle((state[0], state[1]), radius=1, zorder=4, facecolor = facecolor, alpha = alpha)
    # else:
    length = state[3]
    width = state[4]
    heading = state[6]

    vertices = np.array([
        [length / 3, width / 2, 1],
        [length / 2, 0, 1],  # add heading arrow
        [length / 3, -width / 2, 1],
        [-length / 2, -width / 2, 1],
        [-length / 2, width / 2, 1],
    ])

    pose = np.array([
        [np.cos(heading), -np.sin(heading), state[0]],
        [np.sin(heading), np.cos(heading), state[1]],
        [0, 0, 1],
    ])

    vertices_global = np.dot(pose, vertices.T).T[:, :2]
    p = Polygon(vertices_global, facecolor=facecolor, alpha=alpha, zorder=4)
    ax.add_patch(p)


def plot_signal(
    dynamic_map_infos: dict,
    t: int,
    ax: plt.Axes = None,
    linewidth: float = None,
    radius: float = None,
):
    if ax is None:
        ax = plt.gca()

    radius = signal_config['radius'] if radius is None else radius
    linewidth = signal_config['linewidth'] if linewidth is None else linewidth
    
    states_list = dynamic_map_infos['state'][t]
    stop_points_list = dynamic_map_infos['stop_point'][t]

    for states, stop_points in zip(states_list, stop_points_list):
        for signal_state, stop_point in zip(states, stop_points):
            if stop_point[-1] == 0:
                continue

            # override default config
            config = signal_config[signal_state]

            facecolor = config['facecolor']
            edgecolor = config['edgecolor']
            alpha = config['alpha']

            if config['shape'] == 'circle':
                p = Circle(
                    stop_point[:2],
                    radius=radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            elif config['shape'] == 'rectangle':
                p = RegularPolygon(
                    stop_point[:2],
                    radius=radius,
                    numVertices=4,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            elif config['shape'] == 'triangle':
                p = RegularPolygon(
                    stop_point[:2],
                    radius=radius,
                    numVertices=3,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            else:
                warnings.warn(f'Unknown shape {config["shape"]}')
                continue
            ax.add_patch(p)

def plot_map_from_graph(map_graph: PyLaneletMap, map_infos: Dict, if_plot_lane=False, fig=None, ax=None):
    plot_id = []
    if fig is None or ax is None:
        fig, ax = setup_canvas()
    
    polylines = map_infos['all_polylines']
        
    for id, lanelet in map_graph.lanelets.items():
        if map_infos['id_to_type'][id] == 'lane':

            lane = map_infos['lane'][id]
            if if_plot_lane:
                config = lane_config[lane['type']]
                vertex = lanelet.get_bbox_vertices()
                p = Polygon(
                    vertex, facecolor=config['color'], edgecolor=config['color'], linewidth=config['linewidth'],
                    alpha=0.5, zorder=1
                )
                ax.add_patch(p)
                plot_lane(lane, map_infos['all_polylines'], ax, color='xkcd:crimson', linewidth=0.5, linestyle='-')

            # plot left boundary
            for boundary in lane['left_boundaries'] + lane['right_boundaries']:
                feature_id = boundary['feature_id']
                if feature_id in plot_id:
                    continue
                else:
                    plot_id.append(feature_id)
                
                feature_type = map_infos['id_to_type'][feature_id]
                if feature_type == 'road_edge':
                    plot_road_edge(
                        map_infos['road_edge'][feature_id], polylines, ax
                    )  #, range=(boundary['start_index'], boundary['end_index']))
                elif feature_type == 'road_line':
                    plot_road_line(
                        map_infos['road_line'][feature_id], polylines, ax
                    )  #, range=(boundary['start_index'], boundary['end_index']))
                else:
                    print(f'feature type {feature_type} not supported')
        elif map_infos['id_to_type'][id] == 'crosswalk':
            cross_walk = map_infos['crosswalk'][id]
            plot_crosswalk(cross_walk, ax)

    return fig, ax

def plot_map(map_infos: Dict, if_plot_lane=True, map_graph = None, fig = None, ax = None):
    polylines = map_infos['all_polylines']
    if fig is None or ax is None:
        fig, ax = setup_canvas()
    for road_line in map_infos['road_line']:
        plot_road_line(road_line, polylines, ax)

    for road_edge in map_infos['road_edge']:
        plot_road_edge(road_edge, polylines, ax)

    for speed_bump in map_infos['speed_bump']:
        plot_speed_bump(speed_bump, polylines, ax)

    for cross_walk in map_infos['crosswalk']:
        plot_crosswalk(cross_walk, polylines, ax)

    for stop_sign in map_infos['stop_sign']:
        polylines_start, polylines_end = stop_sign['polyline_index']
        points = polylines[polylines_start:polylines_end, :2]
        plot_stop_sign(points, ax)

    if if_plot_lane:
        cmap = plt.cm.get_cmap('hsv', len(map_infos['lane']))
        for i, lane in enumerate(map_infos['lane']):
            plot_lane(lane, polylines, ax)
            if map_graph is not None:
                lanelet = map_graph.lanelets[lane['id']]
                vertex = lanelet.get_bbox_vertices()
                p = Polygon(
                    vertex,
                    facecolor=np.random.rand(3,), 
                    # facecolor='xkcd:light grey', edgecolor='xkcd:light grey',
                    linewidth=0.5,
                    alpha=0.2, zorder=1
                )
                ax.add_patch(p)

    return fig, ax