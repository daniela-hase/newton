import warp as wp

from . import ray, ray_cast
from .ray import MAXVAL


SH_C0 = wp.float32(0.28209479177387814)
SH_C1 = wp.float32(0.4886025119029199)
SH_C2_0 = wp.float32(1.0925484305920792)
SH_C2_1 = wp.float32(-1.0925484305920792)
SH_C2_2 = wp.float32(0.31539156525252005)
SH_C2_3 = wp.float32(-1.0925484305920792)
SH_C2_4 = wp.float32(0.5462742152960396)
SH_C3_0 = wp.float32(-0.5900435899266435)
SH_C3_1 = wp.float32(2.890611442640554)
SH_C3_2 = wp.float32(-0.4570457994644658)
SH_C3_3 = wp.float32(0.3731763325901154)
SH_C3_4 = wp.float32(-0.4570457994644658)
SH_C3_5 = wp.float32(1.445305721320277)
SH_C3_6 = wp.float32(-0.5900435899266435)


GSPLAT_SHAPE_ID = wp.uint32(0xFFFFFFFC)


@wp.func
def ray_gsplat(
    transform: wp.transformf, size: wp.vec3f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f
) -> tuple[wp.float32, wp.float32]:
    """Returns the distance at which a ray intersects with an ellipsoid."""

    # map to local frame
    ray_origin_local, ray_direction_local = ray.map_ray_to_local(transform, ray_origin_world, ray_direction_world)

    # invert size^2
    s = wp.vec3f(ray.safe_div(1.0, size[0] * size[0]), ray.safe_div(1.0, size[1] * size[1]), ray.safe_div(1.0, size[2] * size[2]))

    # (x * ray_direction_local + ray_origin_local)' * diag(1 / size^2) * (x * ray_direction_local + ray_origin_local) = 1
    slvec = wp.cw_mul(s, ray_direction_local)
    a = wp.dot(slvec, ray_direction_local)
    b = wp.dot(slvec, ray_origin_local)
    c = wp.dot(wp.cw_mul(s, ray_origin_local), ray_origin_local) - 1.0

    # solve a * x^2 + 2 * b * x + c = 0
    t, t_hits = ray.ray_compute_quadratic(a, b, c)
    if t < MAXVAL:
        return t_hits[0], t_hits[1]
    return MAXVAL, MAXVAL




@wp.func
def compute_sigma_inv(transform: wp.transformf, scale: wp.vec3f):
    S = wp.mat33f(
        1.0  / (scale[0] * scale[0]), 0.0, 0.0,
        0.0, 1.0  / (scale[1] * scale[1]), 0.0,
        0.0, 0.0, 1.0  / (scale[2] * scale[2])
    )

    R = wp.quat_to_matrix(wp.transform_get_rotation(transform))

    return wp.transpose(R) * S * R


@wp.func
def compute_abc(transform: wp.transformf, sigma_inv: wp.mat33f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f) -> tuple[wp.float32, wp.float32, wp.float32]:
    om = ray_origin_world - wp.transform_get_translation(transform)
    A = wp.dot(ray_direction_world, sigma_inv * ray_direction_world)
    B = wp.dot(ray_direction_world, sigma_inv * om)
    C = wp.dot(om, sigma_inv * om)
    return A, B, C


@wp.func
def ray_gsplat2(transform: wp.transformf, sigma_inv: wp.mat33f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f) -> tuple[wp.float32, wp.float32]:
    k = 1.0

    A, B, C = compute_abc(transform, sigma_inv, ray_origin_world, ray_direction_world)
    C = C - k * k

    disc = B * B - A * C
    if disc < 0:
        return MAXVAL, MAXVAL

    s = wp.sqrt(disc)
    t0 = (-B - s) / A
    t1 = (-B + s) / A
    return t0, t1



@wp.func
def gaussian_mass_along_ray(transform: wp.transformf, opacity: wp.float32, sigma_inv: wp.mat33f, ray_origin_world: wp.vec3f, ray_direction_world: wp.vec3f, t0: wp.float32, t1: wp.float32) -> wp.float32:
    A, B, C = compute_abc(transform, sigma_inv, ray_origin_world, ray_direction_world)

    t_peak = -B / A
    t_peak = wp.clamp(t_peak, t0, t1)

    d2 = A * t_peak * t_peak + 2.0 * B * t_peak + C
    return d2
    peak = opacity * wp.exp(-0.5 * d2)
    return peak

    width = wp.sqrt(2.0 * wp.PI / A)
    seg = wp.max(0.0, t1 - t0)
    return peak * wp.min(width, seg) * 2.0


# def compute_cov_3D(transform: wp.transformf, scale: wp.vec3f) -> wp.mat33f:
#     S = wp.mat33f(
#         scale[0], 0.0, 0.0,
#         0.0, scale[1], 0.0,
#         0.0, 0.0, scale[2]
#     )

#     q = wp.transform_get_rotation(transform)
#     r = q[0]
#     x = q[1]
#     y = q[2]
#     z = q[3]

#     R = wp.mat33f(
#         1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y),
#         2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x),
#         2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)
#     )

#     M = S * R
#     return wp.transpose(M) * M



# def compute_cov_2D(vec4 mean_view, float focal_x, float focal_y, float tan_fovx, float tan_fovy, mat3 cov3D, mat4 viewmatrix) -> wp.vec3f:
# {
#     vec4 t = mean_view;
#     // why need this? Try remove this later
#     float limx = 1.3f * tan_fovx;
#     float limy = 1.3f * tan_fovy;
#     float txtz = t.x / t.z;
#     float tytz = t.y / t.z;
#     t.x = min(limx, max(-limx, txtz)) * t.z;
#     t.y = min(limy, max(-limy, tytz)) * t.z;

#     mat3 J = mat3(
#         focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
# 		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
# 		0, 0, 0
#     );
#     mat3 W = transpose(mat3(viewmatrix));
#     mat3 T = W * J;

#     mat3 cov = transpose(T) * transpose(cov3D) * T;
#     // Apply low-pass filter: every Gaussian should be at least
# 	// one pixel wide/high. Discard 3rd row and column.
# 	cov[0][0] += 0.3f;
# 	cov[1][1] += 0.3f;
#     return vec3(cov[0][0], cov[0][1], cov[1][1]);
# }


@wp.struct
class GSplatShade:
    hit: wp.bool
    color: wp.vec3f
    distance: wp.float32


@wp.func
def get_furthest_hit_distance(bvh_id: wp.uint64, group_root: wp.int32, ray_origin: wp.vec3f, ray_direction: wp.vec3f, transforms: wp.array(dtype=wp.transformf), scales: wp.array(dtype=wp.vec3f), max_distance: wp.float32) -> wp.float32:
    query = wp.bvh_query_ray(bvh_id, ray_origin, ray_direction, group_root)

    hit_index = wp.int32(0)
    furthest_hit_distance = wp.float32(0)
    while wp.bvh_query_next(query, hit_index, max_distance):
        near_hit, far_hit = ray_gsplat(transforms[hit_index], scales[hit_index], ray_origin, ray_direction)
        if far_hit < MAXVAL and far_hit > furthest_hit_distance:
            furthest_hit_distance = far_hit

    return furthest_hit_distance


@wp.func
def get_closest_hit(bvh_id: wp.uint64, group_root: wp.int32, ray_origin: wp.vec3f, ray_direction: wp.vec3f, transforms: wp.array(dtype=wp.transformf), scales: wp.array(dtype=wp.vec3f), max_distance: wp.float32) -> tuple[wp.int32, wp.float32]:
    query = wp.bvh_query_ray(bvh_id, ray_origin, ray_direction, group_root)

    hit_index = wp.int32(0)
    closest_hit = wp.int32(-1)
    while wp.bvh_query_next(query, hit_index, max_distance):
        near_hit, far_hit = ray_gsplat(transforms[hit_index], scales[hit_index], ray_origin, ray_direction)
        if far_hit <= max_distance:
            max_distance = far_hit
            closest_hit = hit_index

    return closest_hit, max_distance

@wp.func
def shade(
    bvh_gsplat_size: wp.int32,
    bvh_gsplat_id: wp.uint64,
    bvh_gsplat_group_roots: wp.array(dtype=wp.int32),
    gsplat_transforms: wp.array(dtype=wp.transformf),
    gsplat_scales: wp.array(dtype=wp.vec3f),
    gsplat_spherical_harmonics: wp.array(dtype=wp.float32, ndim=2),
    gsplat_opacities: wp.array(dtype=wp.float32),
    world_index: wp.int32,
    has_global_world: wp.bool,
    max_distance: wp.float32,
    ray_origin_world: wp.vec3f,
    ray_dir_world: wp.vec3f,
) -> GSplatShade:

    result = GSplatShade()
    result.hit = False
    result.color = wp.vec3f(0.0)
    result.distance = max_distance

    if bvh_gsplat_size:
        for i in range(2 if has_global_world else 1):
            world_index, group_root = ray_cast.get_group_roots(bvh_gsplat_group_roots, world_index, i)
            if group_root < 0:
                continue

            # # max_distance = get_furthest_hit_distance(bvh_gsplat_id, group_root, ray_origin_world, ray_dir_world, gsplat_transforms, gsplat_scales, max_distance) + 0.01
            # max_distance = 10.0

            # ray_origin_inv = ray_origin_world + (ray_dir_world * max_distance)
            # ray_direction_inv = -ray_dir_world

            # count = wp.int32(0)
            # hit_index, hit_distance = get_closest_hit(bvh_gsplat_id, group_root, ray_origin_inv, ray_direction_inv, gsplat_transforms, gsplat_scales, max_distance)
            # while hit_index > -1:
            #     color = SH_C0 * wp.vec3f(gsplat_spherical_harmonics[hit_index][0], gsplat_spherical_harmonics[hit_index][1], gsplat_spherical_harmonics[hit_index][2])
            #     color += wp.vec3f(0.5)

            #     opacity = gsplat_opacities[hit_index]

            #     result.hit = True
            #     result.color = (color * opacity) + (result.color * (1.0 - opacity))

            #     EPSILON = 1e-4
            #     ray_origin_inv += ray_direction_inv * (hit_distance + EPSILON)

            #     hit_index, hit_distance = get_closest_hit(bvh_gsplat_id, group_root, ray_origin_inv, ray_direction_inv, gsplat_transforms, gsplat_scales, max_distance)

            #     # count += 1
            #     # if count > 4:
            #     #     result.color = wp.vec3f(1.0, 0.0, 0.0)
            #     #     break

            query = wp.bvh_query_ray(bvh_gsplat_id, ray_origin_world, ray_dir_world, group_root)
            hit_index = wp.int32(0)

            num_hits = wp.int32(0)
            max_num_hits = 20

            hit_distances = wp.types.vector(max_distance, length=max_num_hits, dtype=wp.float32)
            hit_indices = wp.types.vector(-1, length=max_num_hits, dtype=wp.int32)

            while wp.bvh_query_next(query, hit_index, hit_distances[-1]):
                near_hit, far_hit = ray_gsplat(gsplat_transforms[hit_index], gsplat_scales[hit_index], ray_origin_world, ray_dir_world)

                if near_hit < MAXVAL:
                    if num_hits < max_num_hits:
                        num_hits += 1

                    for h in range(num_hits):
                        if near_hit < hit_distances[h]:
                            for hh in range(num_hits - 1, h, -1):
                                hit_distances[hh] = hit_distances[hh - 1]
                                hit_indices[hh] = hit_indices[hh - 1]
                            hit_distances[h] = near_hit
                            hit_indices[h] = hit_index
                            break

            if num_hits > 0:
                result.hit = True
                for hit in reversed(range(num_hits)):
                    hit_index = hit_indices[hit]

                    color = SH_C0 * wp.vec3f(gsplat_spherical_harmonics[hit_index][0], gsplat_spherical_harmonics[hit_index][1], gsplat_spherical_harmonics[hit_index][2])
                    color += wp.vec3f(0.5)

                    opacity = gsplat_opacities[hit_index]

                    result.hit = True
                    result.color = (color * opacity) + (result.color * (1.0 - opacity))

                    # sigma_inv = compute_sigma_inv(gsplat_transforms[hit_index], gsplat_scales[hit_index])
                    
                    # power = gaussian_mass_along_ray(gsplat_transforms[hit_index], alpha, sigma_inv, ray_origin_world, ray_dir_world, hit_near[hit], hit_far[hit])
                    # power = power * 10.0
                    # power = -0.5 * power
                    # alpha = alpha * wp.exp(power)


                    # color = SH_C0 * wp.vec3f(gsplat_spherical_harmonics[hit_index][0], gsplat_spherical_harmonics[hit_index][1], gsplat_spherical_harmonics[hit_index][2])
                    # color += wp.vec3f(0.5)
                    # result.color = (color * alpha) + (result.color * (1.0 - alpha))
                    # result.color += alpha * color
                    # total *= (1.0 - alpha)

                    # if total < 0.0001:
                    #     break

    return result
