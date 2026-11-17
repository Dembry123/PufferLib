#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
typedef float obs_t;
#include "pufferenv.h"
#define float3 bc_raymath_float3
#include "raymath.h"
#undef float3

#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1}
#define OBS_SIZE 32
#define NUM_ATNS 7
#define G0_NJOINTS 6
#define BC_PI 3.14159265358979323846f
#define BC_DT (1.0f / 60.0f)
#define BC_MAX_GRIP_VEL 2.0f
#define G0_GRIPPER_OPEN_RAD (165.0f * BC_PI / 180.0f)
// A 14-tooth module-2 pinion moves each rack 14 mm per drive radian.
#define G0_RACK_M_PER_RAD 0.014f
// Nominal fingertip and controller tool offsets along wrist +X, in meters.
#define G0_GRIPPER_TIP_X 0.146915092086197f
#define G0_TOOL_X 0.180f

// Gradient0: Z-up, meters/radians, controller soft limits.
// Kinematic prototype: velocity commands, grasp thresholds, proxy torque.
typedef struct {
    float x, y, z;
} G0Vec3;

typedef struct {
    float m[3][3];
    G0Vec3 p;
} G0Frame;

float g0_clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

G0Vec3 g0_xform(G0Frame* f, G0Vec3 v) {
    return (G0Vec3){
        f->m[0][0]*v.x + f->m[0][1]*v.y + f->m[0][2]*v.z + f->p.x,
        f->m[1][0]*v.x + f->m[1][1]*v.y + f->m[1][2]*v.z + f->p.y,
        f->m[2][0]*v.x + f->m[2][1]*v.y + f->m[2][2]*v.z + f->p.z,
    };
}

float g0_clamp_joint(int joint, float q) {
    // J1 uses controller +/-pi; URDF +/-pi/2 remains a hardware-validation issue.
    float lo[] = {-BC_PI, -BC_PI/2, -BC_PI/2, -BC_PI, -1.8326f, -BC_PI};
    float hi[] = {BC_PI, BC_PI/2, BC_PI/2, BC_PI, 2.0944f, BC_PI};
    return g0_clampf(q, lo[joint], hi[joint]);
}

float g0_clamp_gripper(float theta) {
    return g0_clampf(theta, 0, G0_GRIPPER_OPEN_RAD);
}

float g0_gripper_width(float drive_rad) {
    return 2 * G0_RACK_M_PER_RAD * g0_clamp_gripper(drive_rad);
}

float g0_gripper_from_width(float width) {
    return g0_clamp_gripper(width / (2 * G0_RACK_M_PER_RAD));
}

void g0_fk(const float* q, G0Frame* links, G0Vec3* wrist,
    G0Vec3* tool, G0Vec3* tip, G0Frame* wrist_frame) {
    // Joint offsets and axes from the GradientOS mini-6dof-arm model.
    G0Vec3 offsets[] = {{0, 0, 0.0843f}, {0, 0, 0.04315f},
        {0, 0, 0.19715f}, {0.176556f, 0, 0.0455f},
        {0.05515f, 0, 0}, {0.0773f, 0, 0}};
    int axes[] = {2, 1, 1, 0, 1, 0};
    G0Frame T = {.m = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (int i = 0; i < G0_NJOINTS; i++) {
        G0Frame R = {.m = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
        int b = (axes[i] + 1) % 3;
        int c = (axes[i] + 2) % 3;
        R.m[b][b] = R.m[c][c] = cosf(q[i]);
        R.m[b][c] = -sinf(q[i]);
        R.m[c][b] = sinf(q[i]);
        G0Frame next = {.p = g0_xform(&T, offsets[i])};
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                for (int k = 0; k < 3; k++) {
                    next.m[row][col] += T.m[row][k] * R.m[k][col];
                }
            }
        }
        T = next;
        if (links) {
            links[i] = T;
        }
    }
    if (wrist) {
        *wrist = T.p;
    }
    if (wrist_frame) {
        *wrist_frame = T;
    }
    if (tool) {
        *tool = g0_xform(&T, (G0Vec3){G0_TOOL_X, 0, 0});
    }
    if (tip) {
        *tip = g0_xform(&T, (G0Vec3){G0_GRIPPER_TIP_X, 0, 0});
    }
}

typedef struct {
    float start_height, pitch, max_turns;
    float breakaway_torque, running_torque, max_axial_force;
    float radial_tol, bottle_x, bottle_y, bottle_radius;
    int direction;
} HelixParams;

typedef struct {
    float angle, axial;
    int broken_away, detached, stripped;
} HelixState;

// Quasi-static constraint: cap travel cannot exceed the supplied grasp rotation.
// Overload strips the cap and is a failure, not successful unscrewing.
void helix_apply(HelixParams* p, HelixState* s, float delta_angle,
    float torque, float axial_force, float radial_error, int grasped) {
    if (s->detached || !grasped || radial_error > p->radial_tol) {
        return;
    }
    if (fabsf(axial_force) > p->max_axial_force) {
        s->detached = s->stripped = 1;
        return;
    }
    float threshold = s->broken_away ? p->running_torque : p->breakaway_torque;
    if (torque * p->direction <= threshold || delta_angle * p->direction <= 0) {
        return;
    }
    float end = 2 * BC_PI * p->max_turns;
    float progress = fminf((s->angle + delta_angle) * p->direction, end);
    s->angle = p->direction * progress;
    s->axial = p->start_height + p->pitch * progress / (2 * BC_PI);
    s->broken_away = 1;
    s->detached = progress >= end;
}

struct Log {
    float perf, score, episode_return, episode_length;
    float success, detached, cap_turns;
    float max_steps_termination, drop_termination, table_termination;
    float overload_termination, n;
};

struct Env {
    Log log;
    Agent agents[1];
    int tag, boundary_reached, num_agents;
    unsigned int rng;
    float q[G0_NJOINTS], qd[G0_NJOINTS];
    float gripper, gripper_vel;
    G0Vec3 tip;
    G0Frame wrist_frame;
    HelixParams helix_p;
    HelixState helix_s;
    int holding, tick, max_steps;
    float episode_return, max_joint_vel, table_z, grasp_width_max;
};
typedef Env Bottlecap;

void compute_observations(Bottlecap* env) {
    float* o = env->agents[0].observations;
    int i = 0;
    for (int j = 0; j < G0_NJOINTS; j++) {
        o[i++] = env->q[j] / BC_PI;
    }
    for (int j = 0; j < G0_NJOINTS; j++) {
        o[i++] = env->qd[j] / env->max_joint_vel;
    }
    o[i++] = env->tip.x;
    o[i++] = env->tip.y;
    o[i++] = env->tip.z;
    for (int col = 0; col < 2; col++) {
        for (int row = 0; row < 3; row++) {
            o[i++] = env->wrist_frame.m[row][col];
        }
    }
    o[i++] = g0_gripper_width(env->gripper) / g0_gripper_width(G0_GRIPPER_OPEN_RAD);
    o[i++] = env->gripper_vel / BC_MAX_GRIP_VEL;
    o[i++] = env->helix_s.angle / (2 * BC_PI * env->helix_p.max_turns);
    o[i++] = env->helix_s.axial;
    o[i++] = env->helix_s.detached;
    o[i++] = env->holding;
    o[i++] = env->tip.x - env->helix_p.bottle_x;
    o[i++] = env->tip.y - env->helix_p.bottle_y;
    o[i++] = env->tip.z - env->helix_s.axial;
    o[i++] = env->helix_s.broken_away;
    o[i++] = (float)env->tick / env->max_steps;
    assert(i == OBS_SIZE);
}

void puf_reset(Bottlecap* env) {
    env->episode_return = 0;
    env->tick = 0;
    memset(env->qd, 0, sizeof(env->qd));
    env->gripper_vel = 0;
    float q[] = {0, -0.55f, 1.05f, 0, 0.55f, 0};
    memcpy(env->q, q, sizeof(q));
    env->gripper = g0_gripper_from_width(0.028f);
    g0_fk(env->q, NULL, NULL, NULL, &env->tip, &env->wrist_frame);
    env->helix_p.bottle_x = env->tip.x;
    env->helix_p.bottle_y = env->tip.y;
    env->helix_p.start_height = env->tip.z;
    env->helix_s = (HelixState){.axial = env->tip.z};
    env->holding = 1;
    compute_observations(env);
}

void puf_step(Bottlecap* env) {
    float* a = env->agents[0].actions;
    float previous_angle = env->helix_s.angle;
    G0Frame previous = env->wrist_frame;
    for (int j = 0; j < G0_NJOINTS; j++) {
        float q = env->q[j];
        float velocity = g0_clampf(a[j], -1, 1) * env->max_joint_vel;
        env->q[j] = g0_clamp_joint(j, q + velocity * BC_DT);
        env->qd[j] = (env->q[j] - q) / BC_DT;
    }
    float gripper = env->gripper;
    env->gripper = g0_clamp_gripper(gripper
        + g0_clampf(a[6], -1, 1) * BC_MAX_GRIP_VEL * BC_DT);
    env->gripper_vel = (env->gripper - gripper) / BC_DT;
    g0_fk(env->q, NULL, NULL, NULL, &env->tip, &env->wrist_frame);

    // Z component of the small world-frame rotation between consecutive poses.
    // Unlike Euler yaw, this does not jump when the tool passes vertical.
    float delta = 0;
    for (int col = 0; col < 3; col++) {
        delta += previous.m[0][col] * env->wrist_frame.m[1][col]
            - previous.m[1][col] * env->wrist_frame.m[0][col];
    }
    delta *= 0.5f;
    float width = g0_gripper_width(env->gripper);
    float radial = hypotf(env->tip.x - env->helix_p.bottle_x,
        env->tip.y - env->helix_p.bottle_y);
    env->holding = width >= 2 * env->helix_p.bottle_radius - 0.004f
        && width <= env->grasp_width_max && radial <= env->helix_p.radial_tol
        && fabsf(env->tip.z - env->helix_s.axial) <= 0.025f;
    helix_apply(&env->helix_p, &env->helix_s, delta, 0.08f * delta / BC_DT,
        20 * (env->tip.z - env->helix_s.axial), radial, env->holding);

    int table = env->tip.z < env->table_z + 0.01f;
    int drop = !env->holding && !env->helix_s.detached;
    int overload = env->helix_s.stripped;
    int success = env->helix_s.detached && !overload && !table && env->holding;
    int timeout = ++env->tick >= env->max_steps;
    float turns = fabsf(env->helix_s.angle) / (2 * BC_PI);
    float progress = fabsf(env->helix_s.angle) - fabsf(previous_angle);
    float reward = 2 * progress / (2 * BC_PI * env->helix_p.max_turns) - 0.001f
        + 5 * success - 2 * (drop || table || overload);
    env->agents[0].rewards[0] = reward;
    env->episode_return += reward;
    int done = success || drop || table || overload || timeout;
    env->agents[0].terminals[0] = done;
    if (done) {
        env->log.perf += success;
        env->log.score += env->episode_return;
        env->log.episode_return += env->episode_return;
        env->log.episode_length += env->tick;
        env->log.success += success;
        env->log.detached += env->helix_s.detached;
        env->log.cap_turns += turns;
        env->log.max_steps_termination += timeout;
        env->log.drop_termination += drop;
        env->log.table_termination += table;
        env->log.overload_termination += overload;
        env->log.n++;
        puf_reset(env);
    } else {
        compute_observations(env);
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    env->max_steps = dict_get(kwargs, "max_steps");
    env->max_joint_vel = dict_get(kwargs, "max_joint_vel");
    env->table_z = dict_get(kwargs, "table_z");
    env->grasp_width_max = dict_get(kwargs, "grasp_width_max");
    assert(env->max_steps > 0 && env->max_joint_vel > 0);
    assert(env->grasp_width_max >= 0.028f);
    assert(env->grasp_width_max < g0_gripper_width(G0_GRIPPER_OPEN_RAD));
    env->helix_p = (HelixParams){.pitch = 0.003f, .max_turns = 2.5f,
        .breakaway_torque = 0.15f, .running_torque = 0.05f, .max_axial_force = 40,
        .radial_tol = 0.015f, .bottle_radius = 0.015f, .direction = 1};
    env->agents[0].action_mask = NULL;
    env->agents[0].policy = 0;
}

static Model bc_model;
static Shader bc_lighting;
static int bc_view_location;
static Matrix bc_inverse_bind[G0_NJOINTS];
static Vector3 bc_pinion_bind;

Matrix bc_render_pose(G0Frame* f) {
    // Change basis from simulation Z-up to the GLB's Y-up: (x, y, z) -> (x, z, -y).
    return (Matrix){
         f->m[0][0],  f->m[0][2], -f->m[0][1],  f->p.x,
         f->m[2][0],  f->m[2][2], -f->m[2][1],  f->p.z,
        -f->m[1][0], -f->m[1][2],  f->m[1][1], -f->p.y,
        0, 0, 0, 1,
    };
}

void puf_render(Bottlecap* env) {
    // GLB primitive order: base, L1-L5, housing, servo, pinion, and both racks.
    int mesh_link[] = {0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8, 9, 9};
    static float yaw = 0.67f, pitch = 0.36f, distance = 1.06f;
    Camera3D camera = {.target = {0.21f, 0.18f, 0}, .up = {0, 1, 0}, .fovy = 39,
        .projection = CAMERA_PERSPECTIVE};
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(960, 720, "PufferLib Gradient0 Bottlecap");
        SetTargetFPS(60);
        bc_model = LoadModel("resources/bottlecap/gradient0.glb");
        assert(bc_model.meshCount == sizeof(mesh_link) / sizeof(mesh_link[0]));

        // Raylib bakes node transforms into vertices at the GLB's zero pose.
        float zero[G0_NJOINTS] = {0};
        G0Frame bind[G0_NJOINTS];
        g0_fk(zero, bind, NULL, NULL, NULL, NULL);
        for (int i = 0; i < G0_NJOINTS; i++) {
            bc_inverse_bind[i] = MatrixInvert(bc_render_pose(&bind[i]));
        }
        G0Vec3 pinion = g0_xform(&bind[G0_NJOINTS - 1],
            (G0Vec3){0.045115f, 0, -0.0189496006f});
        bc_pinion_bind = (Vector3){pinion.x, pinion.z, -pinion.y};

        const char* vs = "resources/robot_arm/panda_lighting.vs";
        const char* fs = "resources/robot_arm/panda_lighting.fs";
        if (FileExists(vs) && FileExists(fs)) {
            bc_lighting = LoadShader(vs, fs);
            Vector3 light = {-0.42f, 0.82f, -0.38f};
            SetShaderValue(bc_lighting, GetShaderLocation(bc_lighting, "lightDirection"),
                &light, SHADER_UNIFORM_VEC3);
            bc_view_location = GetShaderLocation(bc_lighting, "viewPosition");
            for (int i = 0; i < bc_model.materialCount; i++) {
                bc_model.materials[i].shader = bc_lighting;
            }
        }
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse = GetMouseDelta();
        yaw -= mouse.x * 0.006f;
        pitch = g0_clampf(pitch + mouse.y * 0.006f, 0.08f, 1.45f);
    }
    distance = g0_clampf(distance - GetMouseWheelMove() * 0.08f, 0.45f, 2.5f);
    float flat = distance * cosf(pitch);
    camera.position = (Vector3){camera.target.x + flat * sinf(yaw),
        camera.target.y + distance * sinf(pitch), camera.target.z + flat * cosf(yaw)};
    if (bc_lighting.id) {
        SetShaderValue(bc_lighting, bc_view_location, &camera.position, SHADER_UNIFORM_VEC3);
    }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    BeginMode3D(camera);
    DrawCubeV((Vector3){0.20f, env->table_z - 0.01f, 0},
        (Vector3){0.82f, 0.02f, 0.60f}, (Color){61, 76, 89, 255});
    Vector3 cap = {env->helix_p.bottle_x, env->helix_s.axial,
        -env->helix_p.bottle_y};
    DrawCylinderEx((Vector3){cap.x, env->table_z, cap.z},
        (Vector3){cap.x, env->helix_p.start_height - 0.006f, cap.z},
        0.025f, env->helix_p.bottle_radius, 24, DARKGREEN);
    DrawCylinderEx((Vector3){cap.x, cap.y - 0.006f, cap.z},
        (Vector3){cap.x, cap.y + 0.006f, cap.z}, env->helix_p.bottle_radius,
        env->helix_p.bottle_radius, 24, GOLD);
    DrawSphere((Vector3){cap.x + 0.011f * cosf(env->helix_s.angle),
        cap.y + 0.006f, cap.z - 0.011f * sinf(env->helix_s.angle)}, 0.002f, WHITE);

    G0Frame links[G0_NJOINTS];
    g0_fk(env->q, links, NULL, NULL, NULL, NULL);
    Matrix pose[10] = {MatrixIdentity()};
    for (int i = 0; i < G0_NJOINTS; i++) {
        pose[i + 1] = MatrixMultiply(bc_inverse_bind[i], bc_render_pose(&links[i]));
    }
    Matrix wrist = pose[G0_NJOINTS];
    Vector3 p = bc_pinion_bind;
    Matrix pinion = MatrixMultiply(MatrixTranslate(-p.x, -p.y, -p.z),
        MatrixRotateY(-env->gripper));
    pose[7] = MatrixMultiply(MatrixMultiply(pinion, MatrixTranslate(p.x, p.y, p.z)), wrist);
    float rack = env->gripper * G0_RACK_M_PER_RAD;
    pose[8] = MatrixMultiply(MatrixTranslate(0, 0, -rack), wrist);
    pose[9] = MatrixMultiply(MatrixTranslate(0, 0, rack), wrist);
    for (int i = 0; i < bc_model.meshCount; i++) {
        DrawMesh(bc_model.meshes[i], bc_model.materials[bc_model.meshMaterial[i]],
            pose[mesh_link[i]]);
    }
    EndMode3D();
    DrawText(TextFormat("tick=%d grip=%.1fdeg hold=%d", env->tick,
        env->gripper * 180 / BC_PI, env->holding), 10, 10, 18, WHITE);
    DrawText(TextFormat("tip=(%.3f,%.3f,%.3f)", env->tip.x, env->tip.y, env->tip.z),
        10, 36, 18, SKYBLUE);
    DrawText(TextFormat("cap turns=%.2f axial=%.3f detached=%d",
        env->helix_s.angle / (2 * BC_PI), env->helix_s.axial, env->helix_s.detached),
        10, 62, 18, YELLOW);
    DrawText(TextFormat("return=%.2f stripped=%d", env->episode_return,
        env->helix_s.stripped), 10, 88, 18, RED);
    DrawText("Drag to orbit | Scroll to zoom", 10, GetScreenHeight() - 27, 16, LIGHTGRAY);
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Bottlecap* env) {
    if (IsWindowReady()) {
        UnloadModel(bc_model);
        if (bc_lighting.id) {
            UnloadShader(bc_lighting);
            bc_lighting = (Shader){0};
        }
        CloseWindow();
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "perf", log->perf);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "detached", log->detached);
    dict_set(out, "cap_turns", log->cap_turns);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
    dict_set(out, "drop_termination", log->drop_termination);
    dict_set(out, "table_termination", log->table_termination);
    dict_set(out, "overload_termination", log->overload_termination);
    dict_set(out, "n", log->n);
}
