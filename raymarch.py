import numpy as np
import matplotlib.pyplot as plt
import pdb

NUMBER_OF_STEPS = 32
MINIMUM_HIT_DISTANCE = 0.001
MAXIMUM_TRACE_DISTANCE = 100
LIGHT_POSTION = np.array([3., 0., 3.])
EYE = np.array([0., 0., -5.])
CENTER = np.array([0., 0., 0.])
UP = np.array([1., 0., 0.])


def normalize(a):
    return a/np.linalg.norm(a)


def get_view_matrix():
    f = normalize(CENTER - EYE)
    s = normalize(np.cross(f, UP))
    u = np.cross(s, f)
    z = np.array([0., 0., 0.])
    c = np.array([[0.], [0.], [0.], [1.]])
    vm_temp = np.vstack((f, s, u, z))
    vm = np.append(vm_temp, c, axis=1)
    return vm


def sdf(p, c=np.array([0., 0., 0.]), r=1.):
    return np.linalg.norm(p-c) - r


def get_normal(p, c=np.array([0., 0., 0.])):
    return normalize(p-c)


def ray_march(ro, rd, view_matrix):
    total_distance_traveled = 0

    for ind in range(NUMBER_OF_STEPS):

        # calculate current distance along ray
        # expected: numpy array (3,)
        current_position_local = ro + total_distance_traveled * rd
        current_position_global = np.dot(view_matrix, np.append(current_position_local, np.array([0]), axis=0))
        current_position = current_position_global[:3]

        # get distance to surface
        distance_to_closest = sdf(current_position)

        if distance_to_closest < MINIMUM_HIT_DISTANCE:
            # we hit something!
            # calculate normal at point
            normal_to_point = get_normal(current_position)

            # Introduce defuse lighting
            direction_to_light = normalize(current_position - LIGHT_POSTION)
            diffuse_intensity = np.max([0.0, np.dot(normal_to_point, direction_to_light)])

            # return red for now diffused
            return np.array([1., 0., 0.]) * diffuse_intensity

        if total_distance_traveled > MAXIMUM_TRACE_DISTANCE:
            # we didnt hit anything! End the loop
            break

        # accumulate distance traveled so far
        total_distance_traveled += distance_to_closest

    return np.array([1., 1., 1.])


def generate_rays(nx=400, ny=400, width=1., height=1.):
    ray_x = np.linspace(-width / 2., width / 2., nx)
    ray_y = np.linspace(-height / 2., height / 2., ny)

    return ray_x, ray_y


if __name__ == '__main__':
    ray_x, ray_y = generate_rays()
    ray_image = np.zeros((ray_x.shape[0], ray_y.shape[0], 3))

    view_matrix = get_view_matrix()

    for indx, rx in enumerate(ray_x):
        for indy, ry in enumerate(ray_y):
            ray = np.array([rx, ry, 1.])
            ray_image[indx, indy, :] = ray_march(EYE, ray, view_matrix)

    # ray = np.array([0., 0., 1.])
    # ray_test = ray_march(ro, ray)

    imgplot = plt.imsave('marching_test.png', ray_image)
