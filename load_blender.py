import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

# for the spherical and original camera path
# trans_t = lambda t : torch.Tensor([
#     [1,0,0,0],
#     [0,1,0,3],
#     [0,0,1,t],
#     [0,0,0,1]]).float()

# rot_phi = lambda phi : torch.Tensor([
#     [1,0,0,0],
#     [0,np.cos(phi),-np.sin(phi),0],
#     [0,np.sin(phi), np.cos(phi),0],
#     [0,0,0,1]]).float()

# rot_theta = lambda th : torch.Tensor([
#     [np.cos(th),0,-np.sin(th),0],
#     [0,1,0,0],
#     [np.sin(th),0, np.cos(th),0],
#     [0,0,0,1]]).float()

# for the cylindrical camera path
def trans_t(t, y): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, y],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius, 3)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def pose_cylinder(theta, phi, radius, y):
    c2w = rot_phi(phi/180.*np.pi)
    c2w = trans_t(radius, y) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    #-------------------- CYLINDRICAL MOVEMENT --------------------#
    num_spirals = 5
    frames = 60+1

    # handles translation in the y direction
    y_start = 8.5
    y_end = 1
    y_spiral_step = (y_end - y_start) / num_spirals
    y_step = y_spiral_step / (fps - 1)
    curr_y = y_start

    # handles phi rotation
    phi_start = -30.0
    phi_end = 45.0
    phi_spiral_step = (phi_end - phi_start) / num_spirals
    phi_step = phi_spiral_step / (frames - 1)
    curr_phi = phi_start

    theta_angles = np.linspace(-180, 180, frames)[:-1]
    render_poses = torch.ones(num_spirals * frames-1, 4, 4)

    for i in range(num_spirals):
        for j in range(len(theta_angles)):
            theta = theta_angles[j]
            render_poses[i*(frames-1) + j] = pose_cylinder(theta,
                                                           curr_phi, 3, curr_y)
            curr_y += y_step
            curr_phi += phi_step

    #-------------------- SPHERICAL MOVEMENT --------------------#
    # fps = 20
    # render_poses = torch.ones((fps - 1)**2, 4, 4)

    # phi_start = 0
    # phi_end = -180
    # phi_step = (phi_end - phi_start) / (fps**2)
    # curr_phi = phi_start

    # theta_angles = np.linspace(-180, 180, fps)[:-1]
    # for i in range(fps - 1):
        # for j in range(len(theta_angles)):
            # theta = theta_angles[j]
            # render_poses[i * (fps-1) +
            # j] = pose_spherical(theta, curr_phi, 6.0)
            # curr_phi += phi_step

    #-------------------- ORIGINAL MOVEMENT --------------------#
    # render_poses = torch.stack([pose_spherical(angle, -30.0, 5.0)
    #                            for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(
                img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], i_split
