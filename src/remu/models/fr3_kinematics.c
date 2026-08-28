/* Native libfranka-v9 model-library compatibility shim for the FR3.
 *
 * libfranka 0.15 downloads a shared library from command 11 and expects the
 * O_T_J*, O_J_J*, and Ji_J_J* symbols below. Dynamics are computed locally by
 * libfranka from the URDF; this library supplies poses and Jacobians only.
 */

#include <math.h>

typedef struct {
  double r[9];
  double p[3];
} Transform;

static Transform identity(void) {
  Transform t = {{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}};
  return t;
}

static Transform compose(Transform a, Transform b) {
  Transform out = {{0}, {0, 0, 0}};
  int row, col, k;
  for (row = 0; row < 3; ++row) {
    for (col = 0; col < 3; ++col) {
      for (k = 0; k < 3; ++k) {
        out.r[3 * row + col] += a.r[3 * row + k] * b.r[3 * k + col];
      }
    }
    out.p[row] = a.p[row] + a.r[3 * row] * b.p[0] +
                 a.r[3 * row + 1] * b.p[1] + a.r[3 * row + 2] * b.p[2];
  }
  return out;
}

static Transform translate(double x, double y, double z) {
  Transform t = identity();
  t.p[0] = x;
  t.p[1] = y;
  t.p[2] = z;
  return t;
}

static Transform rotate_x(double angle) {
  Transform t = identity();
  double c = cos(angle), s = sin(angle);
  t.r[4] = c;
  t.r[5] = -s;
  t.r[7] = s;
  t.r[8] = c;
  return t;
}

static Transform rotate_z(double angle) {
  Transform t = identity();
  double c = cos(angle), s = sin(angle);
  t.r[0] = c;
  t.r[1] = -s;
  t.r[3] = s;
  t.r[4] = c;
  return t;
}

static Transform origin(double x, double y, double z, double roll) {
  return compose(translate(x, y, z), rotate_x(roll));
}

static Transform from_column_major(const double value[16]) {
  Transform t;
  int row, col;
  for (row = 0; row < 3; ++row) {
    for (col = 0; col < 3; ++col) {
      t.r[3 * row + col] = value[row + 4 * col];
    }
    t.p[row] = value[row + 12];
  }
  return t;
}

static void to_column_major(Transform t, double out[16]) {
  int row, col;
  for (col = 0; col < 4; ++col) {
    for (row = 0; row < 4; ++row) {
      out[row + 4 * col] = 0.0;
    }
  }
  for (row = 0; row < 3; ++row) {
    for (col = 0; col < 3; ++col) {
      out[row + 4 * col] = t.r[3 * row + col];
    }
    out[row + 12] = t.p[row];
  }
  out[15] = 1.0;
}

static void chain(const double q[7], const double *flange_to_ee,
                  Transform frames[9], double axes[7][3], double points[7][3]) {
  static const double xyz[7][3] = {
      {0, 0, 0.333}, {0, 0, 0},       {0, -0.316, 0}, {0.0825, 0, 0},
      {-0.0825, 0.384, 0}, {0, 0, 0}, {0.088, 0, 0}};
  static const double roll[7] = {
      0, -1.5707963267948966, 1.5707963267948966, 1.5707963267948966,
      -1.5707963267948966, 1.5707963267948966, 1.5707963267948966};
  Transform t = identity();
  int i;
  for (i = 0; i < 7; ++i) {
    t = compose(t, origin(xyz[i][0], xyz[i][1], xyz[i][2], roll[i]));
    points[i][0] = t.p[0];
    points[i][1] = t.p[1];
    points[i][2] = t.p[2];
    axes[i][0] = t.r[2];
    axes[i][1] = t.r[5];
    axes[i][2] = t.r[8];
    t = compose(t, rotate_z(q[i]));
    frames[i] = t;
  }
  frames[7] = compose(t, translate(0, 0, 0.107));
  frames[8] = flange_to_ee ? compose(frames[7], from_column_major(flange_to_ee))
                           : frames[7];
}

static void zero_jacobian(const double q[7], int frame_index,
                          const double *flange_to_ee, double out[42]) {
  Transform frames[9];
  double axes[7][3], points[7][3];
  int column, row, active = frame_index < 7 ? frame_index + 1 : 7;
  chain(q, flange_to_ee, frames, axes, points);
  for (column = 0; column < 7; ++column) {
    for (row = 0; row < 6; ++row) {
      out[row + 6 * column] = 0.0;
    }
    if (column < active) {
      double dx = frames[frame_index].p[0] - points[column][0];
      double dy = frames[frame_index].p[1] - points[column][1];
      double dz = frames[frame_index].p[2] - points[column][2];
      out[6 * column] = axes[column][1] * dz - axes[column][2] * dy;
      out[1 + 6 * column] = axes[column][2] * dx - axes[column][0] * dz;
      out[2 + 6 * column] = axes[column][0] * dy - axes[column][1] * dx;
      out[3 + 6 * column] = axes[column][0];
      out[4 + 6 * column] = axes[column][1];
      out[5 + 6 * column] = axes[column][2];
    }
  }
}

static void body_jacobian(const double q[7], int frame_index,
                          const double *flange_to_ee, double out[42]) {
  Transform frames[9];
  double axes[7][3], points[7][3], spatial[42];
  int column, row;
  chain(q, flange_to_ee, frames, axes, points);
  zero_jacobian(q, frame_index, flange_to_ee, spatial);
  for (column = 0; column < 7; ++column) {
    double v[3], w[3], shifted[3];
    for (row = 0; row < 3; ++row) {
      v[row] = spatial[row + 6 * column];
      w[row] = spatial[row + 3 + 6 * column];
    }
    shifted[0] = v[0] - (frames[frame_index].p[1] * w[2] - frames[frame_index].p[2] * w[1]);
    shifted[1] = v[1] - (frames[frame_index].p[2] * w[0] - frames[frame_index].p[0] * w[2]);
    shifted[2] = v[2] - (frames[frame_index].p[0] * w[1] - frames[frame_index].p[1] * w[0]);
    for (row = 0; row < 3; ++row) {
      out[row + 6 * column] = frames[frame_index].r[row] * shifted[0] +
                              frames[frame_index].r[3 + row] * shifted[1] +
                              frames[frame_index].r[6 + row] * shifted[2];
      out[row + 3 + 6 * column] = frames[frame_index].r[row] * w[0] +
                                  frames[frame_index].r[3 + row] * w[1] +
                                  frames[frame_index].r[6 + row] * w[2];
    }
  }
}

static void pose(const double q[7], int frame_index, const double *flange_to_ee,
                 double out[16]) {
  Transform frames[9];
  double axes[7][3], points[7][3];
  chain(q, flange_to_ee, frames, axes, points);
  to_column_major(frames[frame_index], out);
}

void Ji_J_J1(double out[42]) { const double q[7] = {0}; body_jacobian(q, 0, 0, out); }
void O_J_J1(double out[42]) { const double q[7] = {0}; zero_jacobian(q, 0, 0, out); }

#define DEFINE_JOINT(N, INDEX)                                                   \
  void Ji_J_J##N(const double q[7], double out[42]) { body_jacobian(q, INDEX, 0, out); } \
  void O_J_J##N(const double q[7], double out[42]) { zero_jacobian(q, INDEX, 0, out); }   \
  void O_T_J##N(const double q[7], double out[16]) { pose(q, INDEX, 0, out); }

void O_T_J1(const double q[7], double out[16]) { pose(q, 0, 0, out); }
DEFINE_JOINT(2, 1)
DEFINE_JOINT(3, 2)
DEFINE_JOINT(4, 3)
DEFINE_JOINT(5, 4)
DEFINE_JOINT(6, 5)
DEFINE_JOINT(7, 6)
DEFINE_JOINT(8, 7)

void Ji_J_J9(const double q[7], const double f[16], double out[42]) {
  body_jacobian(q, 8, f, out);
}
void O_J_J9(const double q[7], const double f[16], double out[42]) {
  zero_jacobian(q, 8, f, out);
}
void O_T_J9(const double q[7], const double f[16], double out[16]) {
  pose(q, 8, f, out);
}
