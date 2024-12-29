import argparse
import jax.numpy as np
import numpy as onp
from PIL import Image
import time
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
DISC_OUTER_R = 6 * S_RADIUS

# 35mm sensor
SENSOR_WIDTH = 0.036

# 25mm focal lenth
FOCAL_LENGTH = 0.025

# Camera position
CAMERA_POS = np.array([0.0, -1.5e11, 5e9])


def unit(vecs):
    """Return unit vectors in the same direction as the input vectors."""
    return vecs / np.expand_dims(np.linalg.norm(vecs, axis=1), -1)


class Camera:
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

        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rx), -np.sin(rx)],
                [0.0, np.sin(rx), np.cos(rx)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        R = np.matmul(np.matmul(Rz, Ry), Rx)

        return np.matmul(vels, R)

    def click(self):
        """Click the camera shutter, and return initial position and velocity vectors for all
        rays.
        """
        res_w, res_h = self.res

        vels = []

        # For scaling vectors down from image resolution to sensor size
        scale_ratio = SENSOR_WIDTH / res_w

        # Initiate a ray through each pixel of the image
        for i in range(res_h):
            for j in range(res_w):
                dx = (j - float(res_w) / 2) * scale_ratio
                dz = (i - float(res_h) / 2) * scale_ratio
                vels.append([dx, self.focal_length, dz])

        avels = np.array(vels)

        # Scale velocities to norm to C
        avels = unit(avels) * C

        # Rotate rays to world coordinate frame
        avels = self.rotate_vels(avels)
        apos = np.repeat(self.pos.reshape(1, -1), len(vels), axis=0)

        return (apos, avels)


class Tracer:
    def __init__(self, cam, outpath):
        self.cam = cam
        self.outpath = outpath

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
        """
        # Detect rays that have crossed the x / y plane
        crossed_xy_plane = np.logical_or(
            np.logical_and(pos[:, 2] > 0, new_pos[:, 2] < 0),
            np.logical_and(pos[:, 2] < 0, new_pos[:, 2] > 0),
        )

        # Detect rays that have crossed into a cylinder defined by the
        # accretion disc inner and outer radius
        crossed_disc_column = np.logical_and(
            np.linalg.norm(pos[:, :2], axis=1) > DISC_INNER_R,
            np.linalg.norm(pos[:, :2], axis=1) < DISC_OUTER_R,
        )

        return np.logical_and(crossed_xy_plane, crossed_disc_column)

    def render_image(self, collisions):
        """Produce an image where the pixels corresponding to rays that have collided with the
        accretion disc are white.
        """
        w, h = self.cam.res
        return np.flip(collisions.reshape((h, w)), axis=0).astype(np.uint8) * 254

    def run(self, max_iter=100):
        """Run the main tracer loop."""
        pos, vels = self.cam.click()

        collisions = np.repeat(False, len(pos))

        for n in trange(max_iter):
            new_pos, new_vels = self.tick((pos, vels))

            new_collisions = self.detect_collisions(pos, new_pos)
            if np.any(new_collisions):
                collisions = np.logical_or(collisions, new_collisions)

            pos, vels = new_pos, new_vels
            time.sleep(0.01)

        w, h = self.cam.res
        image = Image.fromarray(onp.array(self.render_image(collisions)))
        image = image.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        image = image.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
        image.save(self.outpath)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", default="out.png")
    parser.add_argument("-x", "--width", type=int, default=4096)
    parser.add_argument("-y", "--height", type=int, default=3072)
    parser.add_argument("-i", "--max-iter", type=int, default=2048)
    args = parser.parse_args()

    cam = Camera(res=(args.width, args.height))
    tracer = Tracer(cam, args.output_path)
    tracer.run(max_iter=args.max_iter)
