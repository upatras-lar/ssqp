import struct
from multiprocessing import resource_tracker, shared_memory

import numpy as np

# Names of the shared memory blocks for the controller's poses and the map
POSES_NAME = "go2_poses"
MAP_NAME = "go2_map"

# Height grid around the robot, MAP_CELLS by MAP_CELLS cells of MAP_RESOLUTION
MAP_RESOLUTION = 0.02
MAP_CELLS = 200

# Mounting of the L1 (utlidar_lidar) in the base frame: the URDF radar joint
# position and the rotation fitted to the robot's cloud, with its pitch
# matched to the floor under the feet
LIDAR_POS = np.array([0.28945, 0.0, -0.046825])
LIDAR_ROT = np.array(
    [
        [-0.9722, -0.0040, 0.2339],
        [-0.0034, 1.0000, 0.0028],
        [-0.2339, 0.0019, -0.9722],
    ]
)

# Time from a sweep's first point until its message arrives. The robot's
# clock jumps, so points are timed from when their message arrives rather
# than by its stamp
CLOUD_DELAY = 0.067
# Longest the mapper waits past a sweep's last point for the poses to cover it
POSE_WAIT = 0.02

# Measurement noise of a point along its ray, and of the ray's direction
RANGE_STD = 0.01
ANGLE_STD = np.radians(0.5)
# Growth of a cell's height uncertainty with time, in m per square root
# second, which lets new points override cells the odometry has drifted from
DRIFT_STD = 0.01
# Points within TOP_LAYER of the highest one in a cell make up its height in
# each sweep, so a cell on an edge takes the top surface
TOP_LAYER = 0.03
# A cell measured this many standard deviations above its height, by at least
# JUMP_POINTS points in a sweep, takes the new height. One measured as far
# below keeps its height, since such points usually graze past an edge,
# unless it is measured below in LOWER_SWEEPS sweeps in a row
OUTLIER_SIGMAS = 3.0
JUMP_POINTS = 2
LOWER_SWEEPS = 3
# Each sweep is moved up or down to match the map where they overlap, the
# median difference over at least ANCHOR_CELLS cells whose height is known to
# ANCHOR_STD. This keeps the map in place while the pose's height drifts, so
# that a drifting pose does not carry the map along with it. The map sits a
# few millimetres above single sweeps, since its cells keep their highest
# returns, so differences within ANCHOR_SLACK are left as they are. Otherwise
# every sweep would raise the map a little
ANCHOR_STD = 0.02
ANCHOR_CELLS = 20
ANCHOR_SLACK = 0.005
# The feet are spheres of FOOT_RADIUS, and the ground under a foot cannot be
# higher than its bottom, give or take FOOT_TOLERANCE. Cells above a foot,
# such as from hits on the robot's own legs, are cleared, and filled from
# their neighbours until they are measured again. They are not lowered to the
# foot in the map, which only seems below the ground when the base estimate is
# off. The shared grid is still set to the bottoms of the feet on the ground,
# up to FOOT_PIN_REACH around them, since filling the cleared cells from their
# neighbours can raise them again, and the walk holds its stance feet on the
# ground it takes. Its plan moves them a little, and false ground above or
# below them right beside them makes it fail
FOOT_RADIUS = 0.022
FOOT_TOLERANCE = 0.02
FOOT_PIN_REACH = 0.02
# Sweeps taken while the robot jumps, or within JUMP_SETTLE after a jump, are
# left out, since the base estimate is poor then
JUMP_SETTLE = 0.5

# Cells further than DESPECKLE_HEIGHT from the median of their neighbours are
# outliers, mostly from a few noisy points, and take the median
DESPECKLE_HEIGHT = 0.015

# Smoothing of the heights the walk takes, whose slopes set the friction
# cones. Cells within SMOOTH_CELLS are averaged, weighted by their distance
# (SMOOTH_STD cells) and by their height difference (SMOOTH_HEIGHT_STD), so
# that noise is evened out but steps stay sharp
SMOOTH_CELLS = 4
SMOOTH_STD = 3.0
SMOOTH_HEIGHT_STD = 0.01

# Cells next to a step of more than EDGE_HEIGHT are edges, and so are unseen
# cells, whose height is only guessed. The distance from each cell to the
# nearest edge is shared along with the heights, up to EDGE_RANGE
EDGE_HEIGHT = 0.02
EDGE_RANGE = 0.08

# Each cell's step height is how far it is above the lowest cell within
# STEP_RANGE, which is shared as well. Unknown cells have none. The ground
# right past a drop is hidden from the sensor, so for these unknown cells are
# taken to be as low as the lowest known cell within STEP_FILL
STEP_RANGE = 0.08
STEP_FILL = 0.15

# Go2 legs in ssqp order (FL, FR, RL, RR): hip positions in the base frame,
# lateral offset of the thigh, and the thigh and calf lengths
FOOT_NAMES = ["FL", "FR", "RL", "RR"]
HIP_POS = np.array(
    [
        [0.1934, 0.0465, 0.0],
        [0.1934, -0.0465, 0.0],
        [-0.1934, 0.0465, 0.0],
        [-0.1934, -0.0465, 0.0],
    ]
)
THIGH_OFFSET = np.array([0.0955, -0.0955, 0.0955, -0.0955])
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

# Regions around the robot whose points are its own body: a box around the
# base, and capsules around the thighs and calves
BODY_BOX = np.array([0.40, 0.17, 0.12])
THIGH_RADIUS = 0.06
CALF_RADIUS = 0.06
# The L1 also returns false points a few centimetres above the ground around
# the feet, where the body hides the ground and nothing measures it again.
# Points within FEET_CLEAR_RADIUS of a foot (horizontally) and more than
# FEET_CLEAR_HEIGHT above its bottom are left out
FEET_CLEAR_RADIUS = 0.15
FEET_CLEAR_HEIGHT = 0.03


def quat_to_rotation(q):
    """Rotation matrices of quaternions (w, x, y, z), one per row of q"""
    q = np.atleast_2d(q)
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def leg_points(joints):
    """Thigh, knee and foot centre of each leg in the base frame

    joints has one row of 12 joint angles (ssqp order) per sample, and the
    result has shape (samples, 4 legs, 3 points, 3)
    """
    q = np.asarray(joints).reshape(-1, 4, 3)
    a, t, c = q[..., 0], q[..., 1], q[..., 2]
    ca, sa = np.cos(a), np.sin(a)

    def leg_frame(v):
        # Rotates (x, y, z) vectors from the thigh frame about the hip x axis
        x, y, z = v
        return np.stack((x, ca * y - sa * z, sa * y + ca * z), axis=-1)

    zero = np.zeros_like(t)
    thigh = HIP_POS + leg_frame((zero, zero + THIGH_OFFSET, zero))
    knee = thigh + leg_frame(
        (-THIGH_LENGTH * np.sin(t), zero, -THIGH_LENGTH * np.cos(t))
    )
    foot = knee + leg_frame(
        (-CALF_LENGTH * np.sin(t + c), zero, -CALF_LENGTH * np.cos(t + c))
    )
    return np.stack((thigh, knee, foot), axis=2)


def segment_distance(p, a, b):
    """Distance from points p to the segments from a to b, all of shape (n, 3)"""
    ab = b - a
    s = np.clip(
        np.sum((p - a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0
    )
    return np.linalg.norm(p - a - s[:, None] * ab, axis=1)


def own_points(points, joints):
    """Mask of the points (base frame) that hit the robot itself

    joints holds the joint angles at each point's time, one row per point
    """
    mask = np.all(np.abs(points) < BODY_BOX, axis=1)
    legs = leg_points(joints)
    for leg in range(4):
        thigh, knee, foot = legs[:, leg, 0], legs[:, leg, 1], legs[:, leg, 2]
        mask |= segment_distance(points, thigh, knee) < THIGH_RADIUS
        mask |= segment_distance(points, knee, foot) < CALF_RADIUS
    return mask


def beside_feet(points, R, joints):
    """Mask of the points (base frame) beside a foot and above its bottom

    R holds the base rotation and joints the joint angles at each point's
    time, one per point
    """
    feet = leg_points(joints)[:, :, 2]
    offsets = np.einsum("nij,nlj->nli", R, points[:, None] - feet)
    beside = np.linalg.norm(offsets[..., :2], axis=2) < FEET_CLEAR_RADIUS
    above = offsets[..., 2] > FEET_CLEAR_HEIGHT - FOOT_RADIUS
    return np.any(beside & above, axis=1)


def open_shared(name, size, create):
    """Shared memory block, created fresh or opened

    Only the creator removes the block. Python registers opened blocks for
    removal at exit too, so they are taken off that list
    """
    if create:
        try:
            shared_memory.SharedMemory(name=name).unlink()
        except FileNotFoundError:
            pass
        return shared_memory.SharedMemory(name=name, create=True, size=size)
    shm = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


def decode_cloud(msg):
    """Points (n, 3) in the sensor frame and their times within the sweep"""
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(-1, msg.point_step)
    offsets = {f.name: f.offset for f in msg.fields}

    def field(name):
        o = offsets[name]
        return raw[:, o : o + 4].copy().view(np.float32)[:, 0].astype(float)

    points = np.stack([field(c) for c in "xyz"], axis=1)
    return points, field("time")


class PoseHistory:
    """Recent base poses and joint angles by time

    Positions and quaternions (w, x, y, z) are in the frame the map is built
    in, the joints are in ssqp order, the stance fields are 1 for each foot
    (in ssqp order) on the ground and 0 otherwise, and the last field is 1
    while the robot jumps and 0 otherwise
    """

    FIELDS = 1 + 3 + 4 + 12 + 4 + 1

    def __init__(self, capacity=2000):
        self.data = np.zeros((capacity, self.FIELDS))
        self.count = 0

    def add(self, t, pos, quat, joints, stance, jumping):
        self.data[self.count % len(self.data)] = np.concatenate(
            ([t], pos, quat, joints, stance, [jumping])
        )
        self.count += 1

    def add_rows(self, rows):
        for row in rows:
            self.data[self.count % len(self.data)] = row
            self.count += 1

    def rows(self, since=0):
        """Records from number since on, oldest first, and the next number"""
        n = len(self.data)
        first = max(since, self.count - n)
        idx = np.arange(first, self.count) % n
        return self.data[idx], self.count

    def sample(self, times):
        """Pose, rotation and joints at the given times, interpolated

        Returns None when the history does not cover them
        """
        rows, _ = self.rows()
        if (
            len(rows) < 2
            or times.min() < rows[0, 0]
            or times.max() > rows[-1, 0]
        ):
            return None
        t = rows[:, 0]
        i = np.clip(np.searchsorted(t, times) - 1, 0, len(t) - 2)
        s = np.clip(
            (times - t[i]) / np.maximum(t[i + 1] - t[i], 1e-9), 0.0, 1.0
        )[:, None]
        a, b = rows[i], rows[i + 1]
        # Quaternions on the same side, then normalized linear interpolation
        qa, qb = a[:, 4:8], b[:, 4:8]
        qb = np.where(np.sum(qa * qb, axis=1, keepdims=True) < 0, -qb, qb)
        q = (1 - s) * qa + s * qb
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        pos = (1 - s) * a[:, 1:4] + s * b[:, 1:4]
        joints = (1 - s) * a[:, 8:20] + s * b[:, 8:20]
        return pos, quat_to_rotation(q), joints


class ElevationMap:
    """Height grid that moves with the robot

    Cells hold a height and its variance. A cell's slot in the arrays is its
    world index modulo MAP_CELLS, so moving the window clears and reuses the
    slots that leave it without moving the others
    """

    def __init__(self):
        n = MAP_CELLS
        self.height = np.full((n, n), np.nan)
        self.var = np.full((n, n), np.inf)
        # Sweeps in a row that measured each cell below its height
        self.lower = np.zeros((n, n), int)
        # World cell index each slot row and column holds
        self.slot_x = np.arange(n)
        self.slot_y = np.arange(n)
        self.origin = np.zeros(2, int)
        self.last_t = None

    def recenter(self, x, y):
        n = MAP_CELLS
        self.origin = (
            np.floor(np.array([x, y]) / MAP_RESOLUTION).astype(int) - n // 2
        )
        for axis, (slots, o) in enumerate(
            ((self.slot_x, self.origin[0]), (self.slot_y, self.origin[1]))
        ):
            wanted = o + (np.arange(n) - o) % n
            moved = slots != wanted
            if moved.any():
                index = (
                    (moved, slice(None)) if axis == 0 else (slice(None), moved)
                )
                self.height[index] = np.nan
                self.var[index] = np.inf
                self.lower[index] = 0
                slots[:] = wanted

    def age(self, t):
        if self.last_t is not None:
            self.var += DRIFT_STD**2 * max(t - self.last_t, 0.0)
        self.last_t = t

    def _slots(self, xy):
        cells = np.floor(xy / MAP_RESOLUTION).astype(int)
        inside = np.all(
            (cells >= self.origin) & (cells < self.origin + MAP_CELLS), axis=1
        )
        return cells % MAP_CELLS, inside

    def add_points(self, points, var):
        """Fuses world points with the given height variances

        Returns the height the points were moved by to match the map
        """
        slots, inside = self._slots(points[:, :2])
        points, var, slots = points[inside], var[inside], slots[inside]
        if len(points) == 0:
            return 0.0
        flat = slots[:, 0] * MAP_CELLS + slots[:, 1]
        top = np.full(MAP_CELLS * MAP_CELLS, -np.inf)
        np.maximum.at(top, flat, points[:, 2])
        keep = points[:, 2] >= top[flat] - TOP_LAYER
        flat, z, w = flat[keep], points[keep, 2], 1.0 / var[keep]
        sum_w = np.zeros(MAP_CELLS * MAP_CELLS)
        sum_wz = np.zeros(MAP_CELLS * MAP_CELLS)
        count = np.zeros(MAP_CELLS * MAP_CELLS, int)
        np.add.at(sum_w, flat, w)
        np.add.at(sum_wz, flat, w * z)
        np.add.at(count, flat, 1)
        cells = np.flatnonzero(sum_w)
        z = sum_wz[cells] / sum_w[cells]

        anchors = self.var.reshape(-1)[cells] < ANCHOR_STD**2
        shift = 0.0
        if np.count_nonzero(anchors) >= ANCHOR_CELLS:
            shift = np.median(
                self.height.reshape(-1)[cells[anchors]] - z[anchors]
            )
            shift = np.sign(shift) * max(abs(shift) - ANCHOR_SLACK, 0.0)

        self._update(cells, z + shift, 1.0 / sum_w[cells], count[cells])
        return shift

    def clear_above(self, points):
        """Clears the cells under the given world points that are above them"""
        slots, inside = self._slots(points[:, :2])
        if not inside.any():
            return
        flat = slots[inside, 0] * MAP_CELLS + slots[inside, 1]
        above = flat[self.height.reshape(-1)[flat] > points[inside, 2]]
        self.height.reshape(-1)[above] = np.nan
        self.var.reshape(-1)[above] = np.inf
        self.lower.reshape(-1)[above] = 0

    def _update(self, cells, z, var_z, points):
        # Fuses one height per cell, measured by the given number of points
        h = self.height.reshape(-1)
        v = self.var.reshape(-1)
        lower = self.lower.reshape(-1)
        old, old_var = h[cells], v[cells]
        new = np.isnan(old)
        diff = z - old
        sigma = np.sqrt(old_var + var_z)
        above = ~new & (diff > OUTLIER_SIGMAS * sigma)
        below = ~new & (diff < -OUTLIER_SIGMAS * sigma)
        lower[cells[below]] += 1
        replace = new | (above & (points >= JUMP_POINTS))
        replace |= below & (lower[cells] >= LOWER_SWEEPS)
        fuse = ~new & ~above & ~below
        lower[cells[replace | fuse]] = 0
        k = old_var[fuse] / (old_var[fuse] + var_z[fuse])
        h[cells[fuse]] = old[fuse] + k * diff[fuse]
        v[cells[fuse]] = (1 - k) * old_var[fuse]
        h[cells[replace]] = z[replace]
        v[cells[replace]] = var_z[replace]

    def grid(self):
        """Heights and variances in window order, and the first cell's centre

        Row i and column j hold the cell centred at (x0 + i r, y0 + j r).
        Unknown heights are nan
        """
        rows = (self.origin[0] + np.arange(MAP_CELLS)) % MAP_CELLS
        cols = (self.origin[1] + np.arange(MAP_CELLS)) % MAP_CELLS
        height = self.height[np.ix_(rows, cols)]
        var = self.var[np.ix_(rows, cols)]
        x0, y0 = (self.origin + 0.5) * MAP_RESOLUTION
        return height, var, x0, y0


def fill_holes(height, default=0.0):
    """Fills unknown cells from their known neighbours, taking the highest

    Holes grow inwards one cell per pass, and cells with no known cell around
    them at all take the default
    """
    unknown = np.isnan(height)
    height = np.where(unknown, -np.inf, height)
    n = height.shape
    while unknown.any():
        padded = np.pad(height, 1, constant_values=-np.inf)
        highest = np.full(n, -np.inf)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                np.maximum(
                    highest,
                    padded[1 + di : 1 + di + n[0], 1 + dj : 1 + dj + n[1]],
                    out=highest,
                )
        reached = unknown & (highest > -np.inf)
        if not reached.any():
            height[unknown] = default
            break
        height[reached] = highest[reached]
        unknown &= ~reached
    return height


# Pairs to sort, in order, for the median of 9 values to end up fifth
MEDIAN9_NETWORK = [
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 1),
    (3, 4),
    (6, 7),
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 3),
    (5, 8),
    (4, 7),
    (3, 6),
    (1, 4),
    (2, 5),
    (4, 7),
    (4, 2),
    (6, 4),
    (4, 2),
]


def median9(grid):
    """Median of each cell and its 8 neighbours, of a grid without holes"""
    n = grid.shape
    padded = np.pad(grid, 1, mode="edge")
    near = [
        padded[1 + di : 1 + di + n[0], 1 + dj : 1 + dj + n[1]]
        for di in (-1, 0, 1)
        for dj in (-1, 0, 1)
    ]
    for a, b in MEDIAN9_NETWORK:
        near[a], near[b] = np.minimum(near[a], near[b]), np.maximum(
            near[a], near[b]
        )
    return near[4]


def despeckle(height):
    """Replaces the outlying cells of a height grid without holes"""
    median = median9(height)
    return np.where(np.abs(height - median) > DESPECKLE_HEIGHT, median, height)


def smooth(height):
    """Edge preserving (bilateral) smoothing of a height grid without holes

    Smooths along x and then along y, which is much faster than over the
    whole window at once and close to it
    """
    k = SMOOTH_CELLS
    for axis in (0, 1):
        padded = np.pad(
            height,
            [(k, k) if a == axis else (0, 0) for a in (0, 1)],
            mode="edge",
        )
        total = np.zeros(height.shape)
        weights = np.zeros(height.shape)
        for d in range(-k, k + 1):
            near = np.take(
                padded, np.arange(k + d, k + d + height.shape[axis]), axis=axis
            )
            w = np.exp(
                -d * d / (2 * SMOOTH_STD**2)
                - (near - height) ** 2 / (2 * SMOOTH_HEIGHT_STD**2)
            )
            total += w * near
            weights += w
        height = total / weights
    return height


def pin_to_feet(height, bottoms, x0, y0):
    """Sets the cells within FOOT_PIN_REACH of each foot (along each axis) to
    its bottom (world points), the lowest where feet overlap, with the grid's
    first cell centred at (x0, y0)
    """
    pinned = np.full(height.shape, np.inf)
    i = np.round((bottoms[:, 0] - x0) / MAP_RESOLUTION).astype(int)
    j = np.round((bottoms[:, 1] - y0) / MAP_RESOLUTION).astype(int)
    reach = round(FOOT_PIN_REACH / MAP_RESOLUTION)
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            ii, jj = i + di, j + dj
            inside = (
                (ii >= 0)
                & (ii < height.shape[0])
                & (jj >= 0)
                & (jj < height.shape[1])
            )
            np.minimum.at(pinned, (ii[inside], jj[inside]), bottoms[inside, 2])
    return np.where(np.isfinite(pinned), pinned, height)


def unseen(height):
    """Mask of the unknown cells with no known cell next to them

    An unknown cell next to a known one is mostly a gap between points, but
    larger unknown areas are mostly hidden from the sensor, like the ground
    just past a drop
    """
    unknown = np.isnan(height)
    n = height.shape
    padded = np.pad(unknown, 1, constant_values=True)
    alone = unknown.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            alone &= padded[1 + di : 1 + di + n[0], 1 + dj : 1 + dj + n[1]]
    return alone


def edge_distance(height, unseen_cells):
    """Distance from each cell's centre to the nearest edge cell's centre

    height has no holes, and unseen_cells marks the cells whose height was
    guessed, which count as edges. Edges lie between an edge cell and its
    neighbour, so they are at least this far. Cells further than EDGE_RANGE
    from any edge get EDGE_RANGE
    """
    n = height.shape
    padded = np.pad(height, 1, mode="edge")
    edges = unseen_cells.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near = padded[1 + di : 1 + di + n[0], 1 + dj : 1 + dj + n[1]]
            edges |= np.abs(near - height) > EDGE_HEIGHT

    k = int(EDGE_RANGE / MAP_RESOLUTION)
    distance = np.full(n, EDGE_RANGE)
    padded = np.pad(edges, k)
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            d = MAP_RESOLUTION * np.hypot(di, dj)
            if d < EDGE_RANGE:
                near = padded[k + di : k + di + n[0], k + dj : k + dj + n[1]]
                distance[near] = np.minimum(distance[near], d)
    return distance


def lowest_near(grid, distance):
    """Lowest value within distance of each cell along both axes"""
    k = int(distance / MAP_RESOLUTION)
    n = grid.shape
    lowest = grid.copy()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k, k)
        padded = np.pad(lowest, pad, mode="edge")
        for d in range(2 * k + 1):
            near = (
                padded[d : d + n[0], :]
                if axis == 0
                else padded[:, d : d + n[1]]
            )
            np.minimum(lowest, near, out=lowest)
    return lowest


def step_heights(height, filled):
    """Height of each known cell above the lowest cell within STEP_RANGE of it

    height has holes and filled is height with its holes filled. Steps are
    taken as the median over each cell and its neighbours, which leaves out
    single cells that are off
    """
    unknown = np.isnan(height)
    low = np.where(
        unknown,
        lowest_near(np.where(unknown, np.inf, filled), STEP_FILL),
        filled,
    )
    steps = np.zeros(height.shape)
    np.subtract(low, lowest_near(low, STEP_RANGE), out=steps, where=~unknown)
    return median9(steps)


class Mapper:
    """Builds the elevation map from LIDAR sweeps and the pose history"""

    def __init__(self, history):
        self.history = history
        self.map = ElevationMap()
        self.feet_since = 0
        # Bottoms of the feet on the ground since the last sweep, as world
        # points, where the shared grid puts the ground around them
        self.stance_bottoms = np.zeros((0, 3))

    def add_sweep(self, points, times):
        """Fuses one sweep of sensor frame points with their absolute times

        Returns the world points it used, or None while the pose history
        does not cover the sweep or the sweep is left out
        """
        sample = self.history.sample(times)
        if sample is None:
            return None
        pos, R, joints = sample
        rows, _ = self.history.rows()
        recent = (rows[:, 0] >= times.min() - JUMP_SETTLE) & (
            rows[:, 0] <= times.max()
        )
        if rows[recent, -1].any():
            return None
        base_points = points @ LIDAR_ROT.T + LIDAR_POS
        keep = ~own_points(base_points, joints) & ~beside_feet(
            base_points, R, joints
        )
        base_points, pos, R = base_points[keep], pos[keep], R[keep]
        world = np.einsum("nij,nj->ni", R, base_points) + pos

        # Height variance from the range noise along the ray and the angle
        # noise across it
        rays = np.einsum("nij,nj->ni", R, base_points - LIDAR_POS)
        ranges = np.linalg.norm(rays, axis=1)
        vertical = (rays[:, 2] / np.maximum(ranges, 1e-6)) ** 2
        var = RANGE_STD**2 * vertical + (ANGLE_STD * ranges) ** 2 * (
            1 - vertical
        )

        self.map.recenter(*pos[-1, :2])
        self.map.age(times[-1])
        shift = self.map.add_points(world, var)
        world[:, 2] += shift

        # The feet since the last sweep bound the ground below them. They
        # move up or down with the sweep, so that they stay in the map's frame
        # rather than the drifting pose's
        rows, self.feet_since = self.history.rows(self.feet_since)
        R = quat_to_rotation(rows[:, 4:8])
        feet = np.einsum("nij,nlj->nli", R, leg_points(rows[:, 8:20])[:, :, 2])
        feet += rows[:, None, 1:4]
        feet[..., 2] += shift - FOOT_RADIUS
        self.map.clear_above(feet.reshape(-1, 3) + [0.0, 0.0, FOOT_TOLERANCE])
        self.stance_bottoms = feet[rows[:, 20:24] > 0.5]
        return world


class SharedPoses:
    """Pose history records shared between processes

    The writer appends records to a ring and then bumps the count, and the
    reader takes the records the writer cannot have been overwriting. The
    writer marks a restart when the frame of its poses starts over, which
    keeps the number of the first record in the new frame
    """

    HEADER = 24

    def __init__(self, name, create=False, capacity=4000):
        size = self.HEADER + capacity * PoseHistory.FIELDS * 8
        self.shm = open_shared(name, size, create)
        self.capacity = (self.shm.size - self.HEADER) // (
            PoseHistory.FIELDS * 8
        )
        self.records = np.ndarray(
            (self.capacity, PoseHistory.FIELDS),
            np.float64,
            self.shm.buf,
            self.HEADER,
        )
        if create:
            struct.pack_into("qqq", self.shm.buf, 0, 0, 0, 0)

    def count(self):
        return struct.unpack_from("q", self.shm.buf, 0)[0]

    def write(self, t, pos, quat, joints, stance, jumping):
        n = self.count()
        self.records[n % self.capacity] = np.concatenate(
            ([t], pos, quat, joints, stance, [jumping])
        )
        struct.pack_into("q", self.shm.buf, 0, n + 1)

    def restart(self):
        # The first record goes in before the restart number changes
        restarts, _ = self.last_restart()
        struct.pack_into("q", self.shm.buf, 16, self.count())
        struct.pack_into("q", self.shm.buf, 8, restarts + 1)

    def last_restart(self):
        """Number of restarts so far, and the first record since the last"""
        restarts = struct.unpack_from("q", self.shm.buf, 8)[0]
        first = struct.unpack_from("q", self.shm.buf, 16)[0]
        return restarts, first

    def read(self, since):
        """Records from number since on that are safe to use, and the next"""
        before = self.count()
        first = max(since, before - self.capacity + 1)
        rows = self.records[np.arange(first, before) % self.capacity].copy()
        after = self.count()
        # Drop the ones the writer may have overwritten while copying
        stale = max(0, after - self.capacity + 1 - first)
        return rows[stale:], before

    def close(self, unlink=False):
        self.records = None
        self.shm.close()
        if unlink:
            self.shm.unlink()


class SharedMap:
    """Latest height grid, edge distances and step heights shared between
    processes

    A sequence number is odd while the grid is being written, so readers
    retry rather than take a half written grid
    """

    HEADER = 32

    def __init__(self, name, create=False):
        grid = MAP_CELLS * MAP_CELLS * 8
        self.shm = open_shared(name, self.HEADER + 3 * grid, create)
        self.heights = np.ndarray(
            (MAP_CELLS, MAP_CELLS), np.float64, self.shm.buf, self.HEADER
        )
        self.edges = np.ndarray(
            (MAP_CELLS, MAP_CELLS),
            np.float64,
            self.shm.buf,
            self.HEADER + grid,
        )
        self.steps = np.ndarray(
            (MAP_CELLS, MAP_CELLS),
            np.float64,
            self.shm.buf,
            self.HEADER + 2 * grid,
        )
        if create:
            struct.pack_into("qddd", self.shm.buf, 0, 0, 0.0, 0.0, 0.0)

    def write(self, heights, edges, steps, x0, y0, t):
        seq = struct.unpack_from("q", self.shm.buf, 0)[0]
        struct.pack_into("q", self.shm.buf, 0, seq + 1)
        self.heights[:] = heights
        self.edges[:] = edges
        self.steps[:] = steps
        struct.pack_into("qddd", self.shm.buf, 0, seq + 2, x0, y0, t)

    def time(self):
        """Time of the latest grid, 0 before one"""
        return struct.unpack_from("qddd", self.shm.buf, 0)[3]

    def read(self):
        """Heights, edge distances, step heights, x0, y0 and time of the
        latest grid, or None before one"""
        for _ in range(10):
            seq, x0, y0, t = struct.unpack_from("qddd", self.shm.buf, 0)
            if seq == 0:
                return None
            if seq % 2:
                continue
            heights = self.heights.copy()
            edges = self.edges.copy()
            steps = self.steps.copy()
            if struct.unpack_from("q", self.shm.buf, 0)[0] == seq:
                return heights, edges, steps, x0, y0, t
        return None

    def close(self, unlink=False):
        self.heights = None
        self.edges = None
        self.steps = None
        self.shm.close()
        if unlink:
            self.shm.unlink()
