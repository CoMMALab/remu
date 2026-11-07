# Supplied YAM model

`yam.urdf` is the Onshape `urdf_top_assembly` XML supplied by the user,
stored unchanged, including its `package://` mesh references. It is the
default model. Its arm joints are `dof_joint1`–`dof_joint6`, and its finger
slides are `dof_joint7`/`dof_joint8`.

No meshes were supplied with the XML. The STL files in `assets/` are reused
from upstream i2rt YAM v1 at revision
`fe78bae67334b9af0b79833dd8ba141a61767f2a`:
https://github.com/i2rt-robotics/i2rt/tree/fe78bae67334b9af0b79833dd8ba141a61767f2a/i2rt/robot_models/arm/yam/v1
The upstream MIT license for those assets is included as `LICENSE`.
For this bundled URDF, the loader maps `meshes/Base.stl` to `assets/base.stl`,
`meshes/Gripper.stl` to `assets/gripper.stl`, and the remaining mesh basenames
to the corresponding files in `assets/`. Exact compatibility with the original
export's unavailable mesh files has not been established.

At import, `dof_joint0` is treated as fixed because its limits are both zero.
Its mounting translation and rotation are retained. The six arm axes, link
frames, inertia tensors and joint limits are read from the supplied XML;
in particular, joint 5 has a negative x axis. RPC angles use these URDF
coordinates directly. Hardware commands require checking coordinate mapping
against the particular leader/driver configuration before connecting it.

Collision geometry is derived from the visual mesh convex
hulls; robot self-collision is disabled. Shared scenes enable cross-arm and
object contact; these remain geometric approximations. The URDF's effort/velocity fields
are export placeholders. The simulator uses the upstream yam_v1.yml arm PD
gains and 10 Nm arm force caps from the upstream YAM MJCF, with an approximate
20 N cap per finger. These are simulation settings, not a calibrated hardware
or motor thermal model. The URDF parallel gripper is not asserted to match
every i2rt linear_4310 hardware revision.

Trajectory limits in `remu.yam.limits` are configurable simulation tuning,
not YAM manufacturer ratings. The defaults use 1.5 rad/s, 10 rad/s²,
100 rad/s³ and 100 Nm/s for arm command filtering. Only the 10 Nm force cap
and the PD gains above have upstream configuration provenance. Default
finger limits are 2 normalized opening units/s, 20 units/s², 200 units/s³,
20 N/finger and 200 N/s/finger. Hardware certification is not implied.
