# -*- coding: utf-8 -*-

import cv2
import numpy as np


class Extractors:

    def __init__(self, surf_param=400, orb_param=10000):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.surf = cv2.xfeatures2d.SURF_create(surf_param)
        self.star = cv2.xfeatures2d.StarDetector_create()
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.orb = cv2.ORB_create(nfeatures=orb_param)
        self.fast = cv2.FastFeatureDetector_create()

    def update(self, surf_param=400, orb_param=10000):
        self.surf = cv2.xfeatures2d.SURF_create(surf_param)
        self.orb = cv2.ORB_create(nfeatures=orb_param)


class Matchers:

    def __init__(self, flann_checks=150, bf_distance=cv2.NORM_L2):
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=flann_checks))
        self.bf = cv2.BFMatcher(bf_distance)

    def update(self, flann_checks=150, bf_distance=cv2.NORM_L2):
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=flann_checks))
        self.bf = cv2.BFMatcher(bf_distance)


class Image:

    def __init__(self, image=None):

        self.keypoints = []
        self.descriptors = []
        self.keypoint_image = None
        self.image = image

    def extract_keypoints(self, extractors, method="fast"):

        if self.image is not None:
            if method == "sift":
                self.keypoints, self.descriptors = extractors.sift.detectAndCompute(self.image, None)
            elif method == "surf":
                self.keypoints, self.descriptors = extractors.surf.detectAndCompute(self.image, None)
            elif method == "brief":
                kp = extractors.star.detect(self.image, None)
                self.keypoints, self.descriptors = extractors.brief.compute(self.image, kp)
            elif method == "orb":
                self.keypoints, self.descriptors = extractors.orb.detectAndCompute(self.image, None)
            elif method == "fast":
                kp = extractors.fast.detect(self.image, None)
                self.keypoints, self.descriptors = extractors.brief.compute(self.image, kp)
            else:
                return False
        else:
            return False

        self.descriptors = np.float32(self.descriptors)

        return True

    def make_keypoint_image(self):

        if self.image is not None:
            self.keypoint_image = cv2.drawKeypoints(self.image, self.keypoints, None, color=(0, 255, 0), flags=0)
            return True

        return False


class StructureFromMotion:

    def __init__(self, K, d):

        self._scale = 1

        self._images = []

        self._K = K
        self._K_inv = np.linalg.inv(K)
        self._d = d

        self._extractors = Extractors()
        self._matchers = Matchers()

        self.points_3d = None

        self.progress = 0.0

    def set_K(self, K):

        self._K = K
        self._K = K / self._scale
        self._K[2] = [0.0, 0.0, 1.0]

        self._K_inv = np.linalg.inv(self._K)

    def set_d(self, d):

        self._d = d

    def set_scale(self, scale):

        self._scale = scale

    def load_images(self, filenames):

        self.progress = 0.0

        for f in filenames:
            try:
                with open(f, 'rb') as file:
                    img = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), 1)
                    img = img[..., ::-1]
                    img = cv2.undistort(img, self._K, self._d)

                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w / self._scale), int(h / self._scale)))

                    self._images.append(Image(img))
            except IOError:
                continue
            finally:
                self.progress += 1.0 / len(filenames)

        self.progress = 1.0

    def remove_images(self, indices):

        self.progress = 0.0

        new_images = []

        for index in range(len(self._images)):
            if index not in indices:
                new_images.append(self._images[index])
            self.progress += 1.0 / len(indices)

        self._images = new_images

        self.progress = 1.0

    def get_images(self):

        return [i.image for i in self._images] if len(self._images) > 0 else []

    def get_keypoint_images(self):

        return [i.keypoint_image for i in self._images] if len(self._images) > 0 else None

    def get_keypoints_number(self):

        return sum([len(i.keypoints) for i in self._images]) if len(self._images) > 0 else 0

    def extract_keypoints(self, method="fast", surf_param=400, orb_param=10000):

        self.progress = 0.0
        i = 1

        self._extractors.update(surf_param, orb_param)

        for img in self._images:
            i += 1
            img.extract_keypoints(self._extractors, method)
            img.make_keypoint_image()

            self.progress += 1.0 / len(self._images)

        self.progress = 1.0

    def match_keypoints(self, kp1, des1, kp2, des2, method="flann", best_percent=0.75):

        if method == "bf":
            matches = self._matchers.bf.match(np.uint8(des1), np.uint8(des2))
        elif method == "flann":
            matches = self._matchers.flann.knnMatch(des1, des2, k=2)
        else:
            return False

        pts1 = []
        pts2 = []

        good = []

        if method == "flann":
            for m, n in matches:
                if m.distance < best_percent * n.distance:
                    pts1.append([kp1[m.queryIdx].pt, kp1[m.queryIdx], des1[m.queryIdx]])
                    pts2.append([kp2[m.trainIdx].pt, kp2[m.trainIdx], des2[m.trainIdx]])
                    good.append([m])
        elif method == "bf":
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:int(len(matches) * best_percent)]
            for m in matches:
                pts1.append([kp1[m.queryIdx].pt, kp1[m.queryIdx], des1[m.queryIdx]])
                pts2.append([kp2[m.trainIdx].pt, kp2[m.trainIdx], des2[m.trainIdx]])
                good.append([m])

        unzip1 = zip(*pts1)
        unzip2 = zip(*pts2)

        pts1, kp1, des1 = np.int32(unzip1[0]), np.asarray(unzip1[1]), np.asarray(unzip1[2])
        pts2, kp2, des2 = np.int32(unzip2[0]), np.asarray(unzip2[1]), np.asarray(unzip2[2])

        return pts1, kp1, des1, pts2, kp2, des2, good

    def find_fundamental_mat(self, kp1, kp2):

        if len(kp1) < 8 and len(kp2) < 8:
            return None, None

        F, F_mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, 0.7, 0.999)

        return F, F_mask

    def find_essential_mat(self, K, F):

        if F is not None and K is not None:
            return K.T.dot(F).dot(K)

        return None

    def find_camera_mat(self, E, K, K_inv, F_mask, pts1, kp1, des1, pts2, kp2, des2):

        if F_mask is None or len(kp1) == 0 and len(kp2) == 0:
            return False

        if E is None:
            E, mask = cv2.findEssentialMat(pts1, pts2, K)

        ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        rt_left = np.hstack((np.eye(3), np.zeros((3, 1))))
        rt_right = np.hstack((R, t))

        left_inliers = []
        right_inliers = []
        left_keypoints = []
        right_keypoints = []
        left_des = []
        right_des = []
        left_pts = []
        right_pts = []

        for i in range(len(pts1)):
            if F_mask[i]:
                left_inliers.append(K_inv.dot([pts1[i][0], pts1[i][1], 1.0]))
                right_inliers.append(K_inv.dot([pts2[i][0], pts2[i][1], 1.0]))

                left_keypoints.append(kp1[i])
                right_keypoints.append(kp2[i])

                left_des.append(des1[i])
                right_des.append(des2[i])

                left_pts.append(pts1[i])
                right_pts.append(pts2[i])

        left_inliers = np.array(left_inliers).reshape(-1, 3)[:, :2]
        right_inliers = np.array(right_inliers).reshape(-1, 3)[:, :2]
        kp1 = np.array(left_keypoints)
        kp2 = np.array(right_keypoints)
        des1 = np.array(left_des)
        des2 = np.array(right_des)
        pts1 = np.array(left_pts)
        pts2 = np.array(right_pts)

        return rt_left, rt_right, left_inliers, right_inliers, pts1, kp1, des1, pts2, kp2, des2

    def triangulate_points(self, rt_left, rt_right, left_inliers, right_inliers):

        points4D = cv2.triangulatePoints(rt_left, rt_right, left_inliers.T, right_inliers.T).T

        points3D = cv2.convertPointsFromHomogeneous(points4D)[:, 0, :]

        return points3D

    def reconstruct_pair(self, img1, img2, match_method="flann", best_percent=0.75):

        K, K_inv = self._K, self._K_inv
        kp1, des1, kp2, des2 = img1.keypoints, img1.descriptors, img2.keypoints, img2.descriptors
        pts1, kp1, des1, pts2, kp2, des2, good = self.match_keypoints(kp1, des1, kp2, des2, match_method, best_percent)
        F, F_mask = self.find_fundamental_mat(pts1, pts2)
        E = self.find_essential_mat(K, F)
        ret = self.find_camera_mat(E, K, K_inv, F_mask, pts1, kp1, des1, pts2, kp2, des2)
        rt_left, rt_right, left_inliers, right_inliers, p1, k1, d1, p2, k2, d2 = ret
        points_3d = self.triangulate_points(rt_left, rt_right, left_inliers, right_inliers)

        return points_3d, rt_right, p2, k2, d2

    def reconstruct_new(self, img, last_pts1, last_kp1, last_des1, points_3d, rt_left, match_method="flann",
                        best_percent=0.75):

        K, K_inv, d = self._K, self._K_inv, self._d
        kp2, des2 = img.keypoints, img.descriptors
        pts1, kp1, des1, pts2, kp2, des2, _ = self.match_keypoints(last_kp1, last_des1, kp2, des2, match_method,
                                                                   best_percent)
        F, F_mask = self.find_fundamental_mat(pts1, pts2)
        E = self.find_essential_mat(K, F)
        ret = self.find_camera_mat(E, K, K_inv, F_mask, pts1, kp1, des1, pts2, kp2, des2)
        rt_l, rt_r, left_inliers, right_inliers, p1, k1, d1, p2, k2, d2 = ret

        match_kp = []
        match_p3d = []

        aset = set([tuple(x) for x in p1])
        bset = set([tuple(x) for x in last_pts1])
        res = np.array([x for x in aset & bset])
        for r in res:
            i = np.where(np.all(p1 == r, axis=1))[0][0]
            j = np.where(np.all(last_pts1 == r, axis=1))[0][0]

            match_kp.append(p1[i])
            match_p3d.append(points_3d[j])

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(match_p3d, dtype=np.float64),
                                                      np.array(match_kp, dtype=np.float64),
                                                      K, d)
        rt_right = np.hstack((cv2.Rodrigues(rvec)[0], tvec))
        points_3d = self.triangulate_points(rt_left, rt_right, left_inliers, right_inliers)

        return points_3d, rt_right, p2, k2, d2

    def reconstruct(self, flann_checks=150, bf_distance=cv2.NORM_L2, match_method="flann", best_percent=0.75):

        if len(self._images) < 2:
            return None

        self._matchers.update(flann_checks, bf_distance)

        pts, kp, des, rt, points_3d, = [None] * 5

        self.progress = 0.0

        for i in range(1, len(self._images)):

            if points_3d is None:
                points_3d, rt, pts, kp, des = self.reconstruct_pair(self._images[0], self._images[1], match_method,
                                                                    best_percent)

                colors = [self._images[1].image[k[1], k[0]] / 255.0 for k in pts]
                points_3d = np.hstack((points_3d, colors))
            else:
                ret = self.reconstruct_new(self._images[i], pts, kp, des, points_3d[:, :3], rt, match_method,
                                           best_percent)
                new_points_3d, rt, pts1, kp1, des1 = ret
                pts = np.concatenate((pts, pts1), axis=0)
                kp = np.concatenate((kp, kp1), axis=0)
                des = np.concatenate((des, des1), axis=0)
                colors = [self._images[1].image[k[1], k[0]] / 255.0 for k in pts1]
                new_points_3d = np.hstack((new_points_3d, colors))
                points_3d = np.concatenate((points_3d, new_points_3d), axis=0)

            self.progress += 1.0 / len(self._images)

        self.progress = 1.0

        self.points_3d = points_3d

        return points_3d
