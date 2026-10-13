"""Elevation map of the terrain around the Go2 from its L1 LIDAR

The controller runs this script as a process of its own, so that mapping
does not hold up its threads. It builds the map from rt/utlidar/cloud and the
controller's state estimate (shared memory), and shares the latest height grid
back through shared memory. It stops when terminated or when the controller
exits

  python lidar_map.py interface [--view] [--save file.rrd]
"""

import argparse
import os
import queue
import signal
import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

import elevation_map as em


class Viewer:
    """Logs the cloud, the map and the robot to Rerun"""

    def __init__(self, spawn, save):
        import rerun as rr

        self.rr = rr
        rr.init("go2_elevation_map", spawn=spawn)
        if save:
            rr.save(save)
        n = em.MAP_CELLS
        i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
        a = (i * n + j).ravel()
        self.triangles = np.concatenate([np.c_[a, a + n, a + 1], np.c_[a + 1, a + n, a + n + 1]])

    def log(self, t, world, height, filled, x0, y0, history):
        rr = self.rr
        rr.set_time("time", timestamp=t)
        z = world[:, 2]
        rr.log("world/points", rr.Points3D(world, radii=0.006, colors=self._colors(z, z.min(), z.max())))

        n = em.MAP_CELLS
        xs = x0 + em.MAP_RESOLUTION * np.arange(n)
        ys = y0 + em.MAP_RESOLUTION * np.arange(n)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        vertices = np.c_[gx.ravel(), gy.ravel(), filled.ravel()]
        known = ~np.isnan(height.ravel())
        lo, hi = np.percentile(filled, [1, 99])
        colors = self._colors(filled.ravel(), lo, hi + 1e-3)
        colors[~known] = [110, 110, 110]
        rr.log(
            "world/map",
            rr.Mesh3D(vertex_positions=vertices, triangle_indices=self.triangles, vertex_colors=colors),
        )

        rows, _ = history.rows(max(history.count - 1, 0))
        if len(rows):
            row = rows[-1]
            w, x, y, zq = row[4:8]
            rr.log("world/base", rr.Transform3D(translation=row[1:4], quaternion=[x, y, zq, w]))
            rr.log("world/base/body", rr.Boxes3D(half_sizes=[0.19, 0.047, 0.057], colors=[220, 220, 220]))
            legs = em.leg_points(row[8:20])[0]
            hips = em.HIP_POS
            strips = [np.vstack((hips[k], legs[k])) for k in range(4)]
            rr.log("world/base/legs", rr.LineStrips3D(strips, radii=0.012, colors=[200, 200, 200]))

    @staticmethod
    def _colors(values, lo, hi):
        # Blue to yellow by value
        s = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)[:, None]
        low, high = np.array([40, 60, 170]), np.array([250, 220, 40])
        return ((1 - s) * low + s * high).astype(np.uint8)


def run_mapper(interface, view=False, save=None):
    """Maps until terminated or until the process that started it exits

    The controller's poses block (em.POSES_NAME) must exist already. view
    opens a Rerun viewer, and save records to that .rrd file
    """
    # The controller stops it, so leave Ctrl-C to the controller, and leave
    # through the cleanup below when terminated
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    parent = os.getppid()

    ChannelFactoryInitialize(0, interface)

    poses = em.SharedPoses(em.POSES_NAME)
    poses_since = 0
    restarts = None
    shared = em.SharedMap(em.MAP_NAME, create=True)
    viewer = Viewer(view, save) if view or save else None

    clouds = queue.Queue(maxsize=8)

    def on_cloud(msg):
        try:
            clouds.put_nowait((time.time(), msg))
        except queue.Full:
            pass

    cloud_sub = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
    cloud_sub.Init(on_cloud, 10)

    try:
        # The parent changes once the controller has exited
        while os.getppid() == parent:
            try:
                received, msg = clouds.get(timeout=0.5)
            except queue.Empty:
                continue
            # The map starts over with the controller's frame
            latest, first = poses.last_restart()
            if latest != restarts:
                restarts = latest
                history = em.PoseHistory()
                mapper = em.Mapper(history)
                poses_since = max(poses_since, first)
            rows, poses_since = poses.read(poses_since)
            history.add_rows(rows)
            points, offsets = em.decode_cloud(msg)
            times = received - em.CLOUD_DELAY + offsets
            world = mapper.add_sweep(points, times)
            if world is None:
                continue
            height, _, x0, y0 = mapper.map.grid()
            known = height[~np.isnan(height)]
            filled = em.fill_holes(height, np.median(known) if known.size else 0.0)
            filled = em.smooth(em.despeckle(filled))
            edges = em.edge_distance(filled, em.unseen(height))
            steps = em.step_heights(height, filled)
            shared.write(filled, edges, steps, x0, y0, received)
            if viewer is not None:
                viewer.log(received, world, height, filled, x0, y0, history)
    finally:
        cloud_sub.Close()
        poses.close()
        shared.close(unlink=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Go2 elevation map from the L1 LIDAR")
    parser.add_argument("interface", help="network interface")
    parser.add_argument("--view", action="store_true", help="show the map in a Rerun viewer")
    parser.add_argument("--save", help="save the map to this Rerun .rrd file")
    args = parser.parse_args()
    run_mapper(args.interface, args.view, args.save)
