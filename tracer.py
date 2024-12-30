"""This module implements a very simple black hole raytracer / simulator"""

import argparse

import jax.numpy as np
import jax.random as jr
import numpy as onp
from PIL import Image
from tqdm import trange

# Physical constants
C = 3e8
G = 6.674e-11
SOLAR_MASS = 1.988e30

# Black hole position, mass and Schwarzschild radius
BH_POS = np.array([0.0, 0.0, 0.0])
BH_MASS = 4e6 * SOLAR_MASS
S_RADIUS = (2 * G * BH_MASS) / C**2

# Accretion disk size
DISC_INNER_R = 3 * S_RADIUS
DISC_OUTER_R = 7.5 * S_RADIUS

# 35mm sensor
SENSOR_WIDTH = 0.036

# 25mm focal lenth
FOCAL_LENGTH = 0.025

# Camera position
CAMERA_POS = np.array([0.0, -1.5e11, 10e9])

# Random number generator
RNG_KEY = jr.PRNGKey(0)

# Gaussian normalization factor
SQRT_2_PI = np.sqrt(2 * np.pi)


def normal(x, sigma):
    """Return normal probability for zero-centered x, at sigma"""
    return 1 / (sigma * SQRT_2_PI) * np.exp(-0.5 * (x**2 / sigma**2))


def unit(vecs):
    """Return unit vectors in the same direction as the input vectors."""
    return vecs / np.expand_dims(np.linalg.norm(vecs, axis=1), -1)


class Camera:
    """The camera class encapsulates attributes of a camera, e.g. its position
    and focal length, and produces the initial tensor of ray positions and
    velocities.
    """

    def __init__(
        self,
        res,
        pos=CAMERA_POS,
        focal_length=FOCAL_LENGTH,
        subject_pos=BH_POS,
        rotation=np.pi / 16,
    ):
        self.res = res
        self.pos = pos
        self.focal_length = focal_length
        self.subject_pos = subject_pos
        self.rotation = rotation

    def rotate_vels(self, vels):
        """Rotate the ray velocity vectors into the world coordinate frame."""
        pos = self.pos - self.subject_pos
        pos_x, pos_y, pos_z = pos

        # Camera always points at origin
        rx = -np.arctan(pos_z / pos_y)
        rz = -np.arctan(pos_x / pos_y)
        ry = self.rotation

        rx_mat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rx), -np.sin(rx)],
                [0.0, np.sin(rx), np.cos(rx)],
            ]
        )

        ry_mat = np.array(
            [
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)],
            ]
        )

        rz_mat = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        r_mat = np.matmul(np.matmul(rz_mat, ry_mat), rx_mat)
        return np.matmul(vels, r_mat)

    def click(self):
        """Click the camera shutter, and return initial position and velocity vectors for all
        rays.
        """
        res_w, res_h = self.res

        # For scaling vectors down from image resolution to sensor size
        scale_ratio = SENSOR_WIDTH / res_w

        # Initiate a ray through each pixel of the image
        j, i = np.meshgrid(np.arange(res_w), np.arange(res_h))
        dx = (j - float(res_w) / 2) * scale_ratio
        dz = (i - float(res_h) / 2) * scale_ratio
        avels = np.stack(
            [
                dx.flatten(),
                np.full(res_w * res_h, self.focal_length),
                dz.flatten(),
            ],
            axis=1,
        )

        # Scale velocities to norm to C
        avels = unit(avels) * C

        # Rotate rays to world coordinate frame
        avels = self.rotate_vels(avels)
        apos = np.repeat(self.pos.reshape(1, -1), len(avels), axis=0)

        return (apos, avels)


class Tracer:
    """The tracer class manages the ray-tracing process."""

    def __init__(self, camera, outpath):
        self.cam = camera
        self.outpath = outpath
        self.has_collisions = False
        self.collision_counts = []
        self.reset()

    def reset(self):
        """Reset the state of the tracer."""
        self.has_collisions = False
        self.collision_counts = []

    def tick(self, rays):
        """Move the rays forward by a single timeslice."""
        pos, vels = rays

        # Update positions
        p = pos + vels

        # Update velocities, according to a = GM/r^2. The direction of acceleration
        # is just the negative of ray's position, since the black hole is at origin and
        # acceleration is always toward the black hole
        r = np.linalg.norm(p, axis=1).reshape(-1, 1)
        v = vels - unit(p) * (G * BH_MASS) / r**2

        # Light can accelerate by changing direction, but always travels at C
        v = unit(v) * C

        return (p, v)

    def detect_collisions(self, pos, new_pos):
        """Detect collisions with the accretion disc. This is greatly simplified by placing the
        black hole at origin, and having the accretion disc sit on the x / y plane in the world
        coordinate frame. In this case, we only need to check for rays that have:

        1) crossed the x / y plane and
        2) crossed within a cylinder defined by the inner and outer radius of the disc.
        3) sampled from a guassian concentrated around the inner side of the disc.
        """
        # Detect rays that have crossed the x / y plane
        crossed_xy_plane = np.logical_or(
            np.logical_and(pos[:, 2] > 0, new_pos[:, 2] < 0),
            np.logical_and(pos[:, 2] < 0, new_pos[:, 2] > 0),
        )

        # Detect rays that have crossed into a cylinder defined by the
        # accretion disc inner and outer radius
        distances = np.linalg.norm(new_pos[:, :2], axis=1)
        crossed_disc_column = np.logical_and(
            distances > DISC_INNER_R,
            distances < DISC_OUTER_R,
        )

        # Produce a probability of collision as a function of the distance. This
        # produces an effect where the mass of the accretion disk is concentrated
        # around the event horizon
        distances = distances - DISC_INNER_R
        disc_dist = DISC_OUTER_R - DISC_INNER_R
        distances = np.clip(distances, 0, disc_dist)
        normalized_distances = distances / disc_dist
        probabilities = normal(normalized_distances, 0.25)
        collision_samples = jr.bernoulli(RNG_KEY, probabilities).astype(bool)

        return np.logical_and(
            crossed_xy_plane,
            np.logical_and(crossed_disc_column, collision_samples),
        )

    def should_stop(self):
        """Stopping criterion for determining when tracing is done."""
        lookback = 10

        if not self.has_collisions or len(self.collision_counts) < lookback:
            return False

        return sum(self.collision_counts[-lookback:]) == 0

    def render_image(self, collisions):
        """Produce an image where the pixels corresponding to rays that have collided with the
        accretion disc are white.
        """
        w, h = self.cam.res
        image = np.flip(collisions.reshape((h, w)), axis=0)
        image = (image / image.max()) * 254
        image = onp.array(image.astype(np.uint8))
        return Image.fromarray(image)

    def antialias_image(self, image):
        """Runs an anti-aliasing process on the image"""
        w, h = self.cam.res
        image = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        image = image.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
        return image

    def run(self, max_iter=100):
        """Run the main tracer loop."""
        self.reset()
        pos, vels = self.cam.click()

        collisions = np.repeat(False, len(pos))

        i = 0
        for _ in trange(max_iter):
            new_pos, new_vels = self.tick((pos, vels))

            new_collisions = self.detect_collisions(pos, new_pos)
            self.collision_counts.append(new_collisions.sum())

            if np.any(new_collisions):
                self.has_collisions = True
                collisions = np.logical_or(collisions, new_collisions)

            if self.should_stop():
                break

            pos, vels = new_pos, new_vels
            i += 1

        print(f"Finished in {i} iterations")

        image = self.render_image(collisions)
        image = self.antialias_image(image)
        image.save(self.outpath)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", default="out.png")
    parser.add_argument("-x", "--width", type=int, default=1920)
    parser.add_argument("-y", "--height", type=int, default=1080)
    parser.add_argument("-i", "--max-iter", type=int, default=1000)
    args = parser.parse_args()

    cam = Camera(res=(args.width, args.height))
    tracer = Tracer(cam, args.output_path)
    tracer.run(max_iter=args.max_iter)
