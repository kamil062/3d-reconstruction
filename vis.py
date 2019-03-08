# -*- coding: utf-8 -*-

import numpy as np
import vispy
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from vispy.scene import visuals


class OutliersDetection:

    @staticmethod
    def find_IQR(points, Q1=0, Q3=90):
        avg = np.average(points[:, :3], axis=0)
        distances = [np.linalg.norm(a - avg) for a in points[:, :3]]

        Q_lower = np.percentile(distances, Q1, interpolation='lower')
        Q_higher = np.percentile(distances, Q3, interpolation='lower')
        IQR = Q_higher - Q_lower
        low = Q_lower - 1.5 * IQR
        high = Q_higher + 1.5 * IQR

        inliers = points[np.argwhere((distances > low) & (distances < high))]
        inliers = inliers.reshape(len(inliers), 6)

        return inliers

    @staticmethod
    def find_LOF(points, contamination=0.1, n_neighbors=20):
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, metric='euclidean')
        y_pred = clf.fit_predict(points[:, :3])

        return points[y_pred == 1]

    @staticmethod
    def find_isolation_forest(points, contamination=0.1):
        clf = IsolationForest(contamination=contamination)

        clf.fit(points[:, :3])
        scores_pred = clf.decision_function(points[:, :3])
        y_pred = clf.predict(points[:, :3])

        return points[y_pred == 1]

    @staticmethod
    def find_elliptic_envelope(points, contamination=0.1):
        clf = EllipticEnvelope(contamination=contamination)

        clf.fit(points[:, :3])
        scores_pred = clf.decision_function(points[:, :3])
        y_pred = clf.predict(points[:, :3])

        return points[y_pred == 1]


class Visualization:

    def __init__(self, points_3d=None):

        if points_3d is not None:
            self._points_3d = points_3d
        else:
            self._points_3d = None

    def set_points(self, points_3d):

        self._points_3d = points_3d

    @staticmethod
    def ply_save(filename, points):

        with open(filename, "w") as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (len(points)))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')

            for x, y, z, r, g, b in points:
                f.write('%f %f %f %d %d %d\n' % (x, y, z, r, g, b))

    def plot_point_cloud(self):

        self._canvas = vispy.scene.SceneCanvas(bgcolor=(0, 0, 0, 0), dpi=200, title="Wizualizacja", show=True)
        self._view = self._canvas.central_widget.add_view()

        points = self._points_3d[:, :3]
        colors = self._points_3d[:, 3:]

        scatter = vispy.scene.visuals.Markers()
        scatter.set_data(points, edge_color=None, face_color=colors, size=5)
        self._view.add(scatter)

        self._view.camera = 'arcball'
        vispy.scene.visuals.XYZAxis(parent=self._view.scene)

        self._canvas.app.run()
        self._canvas.show()
