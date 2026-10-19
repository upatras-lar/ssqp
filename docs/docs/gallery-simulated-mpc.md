# Gallery

<div class="gallery-switch" markdown>

[Trajectory generation](gallery.md){ .md-button }
[Simulated MPC](gallery-simulated-mpc.md){ .md-button .md-button--primary }
[Real MPC](gallery-real-mpc.md){ .md-button }

</div>

Closed-loop [MPC](concepts/mpc.md) running in [MuJoCo](https://mujoco.org/), from the scripts under
[`examples/mujoco_mpc/`](https://github.com/upatras-lar/ssqp/tree/main/examples/mujoco_mpc). Unlike the
[trajectory generation](gallery.md) results, nothing here is solved once and played back: the state comes from the
simulator, SSQP re-solves a short horizon at every control step, and only the first control is applied.

## Pendubot Swing-Up

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/pendubot_mujoco_mpc.mp4" type="video/mp4">
</video>

The pendubot, a double pendulum with only its shoulder joint actuated, swinging up and balancing upright. Produced by
[`pendubot.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/pendubot.py), which builds its model from SSQP's double pendulum preset and tracks the tip
position with a terminal cost on the upright configuration. The script also takes `--noise` to disturb the system
with random torques, and `--identification` to estimate the second link's parameters online. The offline version of
this problem is covered in [Pendubot Swing-Up](examples/pendubot.md).

## Panda Obstacle Avoidance

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/panda_mujoco_mpc.mp4" type="video/mp4">
</video>

A Franka Panda reaching a new random goal every few seconds, drawn as a red sphere, while a random pillar placed
between the end effector and the goal blocks the straight path. Produced by
[`panda_move.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/panda/panda_move.py). A box-shaped bound is convex, so the pillar is not excluded directly:
any stage the previous plan puts over the pillar has to stay above its top face, so the end effector passes over the
pillar rather than around it. This is not a hard guarantee, since the solve has a limited iteration budget. The
closely related [Panda Circle (MPC)](examples/panda-circle-mpc.md) is covered in detail.

## Go2 Standing on Three Legs

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 420px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/go2_standing_mujoco_mpc.webm" type="video/webm">
</video>

The Go2 standing on three legs, with a fourth foot lifted, while pushes of increasing force are applied to it in the
viewer. The MPC holds the base over the support triangle with contact constraints on the three stance feet, and
recovers after each push. Produced by [`go2_standing.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/go2/go2_standing.py) with `--legs 3`; `--legs 4` and
`--legs 2` stand on four and two legs. If a push is hard enough to lift or slip a stance foot, the plan still assumes
that contact, and the optimization soon fails.

## Go2 Walking

<video autoplay loop muted playsinline style="display: block; margin: 0 auto; width: 520px; max-width: 100%; border-radius: 8px;">
  <source src="../assets/go2_walking_mujoco_mpc.webm" type="video/webm">
</video>

A gamepad-driven trot, showing the walk itself, jumping in place, turning the base's orientation while the feet stay
planted and while walking, and changes to the base height and the feet's swing height. Produced by
[`go2_walking.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/mujoco_mpc/go2/go2_walking.py), covered in detail in [Go2 Walking (MPC)](examples/go2-walking-mpc.md).
All of these are changes to the references of a single MPC problem, solved once per 24 ms plan stage with the
whole-body dynamics.
