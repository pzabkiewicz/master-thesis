from abc import ABC, abstractmethod
from random import random

import numpy as np
from math import sqrt, sin, cos
from skimage.feature import local_binary_pattern
from skimage.filters import sobel_h, sobel_v, threshold_otsu
from skimage.morphology import skeletonize

CHAR_PIXEL_VALUE = 1
BACKGROUND_VALUE = 0


class FeatureDescriptor(ABC):

    @abstractmethod
    def __init__(self):
        self.image = None
        self.feature_vector = None

    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def describe(self):
        pass

    @abstractmethod
    def get_feature_vector(self):
        pass


class LocalBinaryPattern(FeatureDescriptor):
    """ This class defines algorithm for preprocessing an input image
    in gray-scale and describing it using Local Binary Pattern algorithm
    in gray-scale and rotation invariant version, invented by T. Olaja et al.

    References:

    [1] T. Olaja, M. Pietikainen, T. Maenpaa, Multiresolution Gray-Scale and Rotation
    Invariant Texture Classification with Local Binary Pattern
    [2] https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/ """

    def __init__(self, n_points=24, radius=3):
        super().__init__()
        self.n_points = n_points
        self.radius = radius

    def preprocess(self, image):
        self.image = image[:, :]

    def describe(self, eps=1e-7):
        self.image = local_binary_pattern(self.image, self.n_points,
                                          self.radius, method='uniform')

        (hist, _) = np.histogram(self.image.ravel(),
                                 bins=np.arange(0, self.n_points + 3),
                                 range=(0, self.n_points + 2))

        # normalize histogram that it sums to 1
        hist = hist.astype('float')
        hist /= (hist.sum() + eps)

        self.feature_vector = hist

    def get_feature_vector(self):
        return self.feature_vector.tolist()


class Zoning(FeatureDescriptor):
    """ This feature descriptor is inspired by an idea of splitting
    the image matrix into zones and the zones into horizontal subzones.
    Then amount of foreground pixels in each subzone is counted.
    Last step is to calculate average from subzones for each zone
    what creates feature vector.

    References:

    [1] S.H. Tanvir, T.A. Khan, A.B. Yamin, Evaluation of Optical Character
    Recognition Algorithms and Feature Extraction Techniques, August 2016 """

    def __init__(self, zone_number=16):
        super().__init__()
        self.side_zone_number = int(sqrt(zone_number))

    def preprocess(self, image):
        thresh = threshold_otsu(image)
        img = image > thresh
        self.image = img[:, :]

    def describe(self):
        side_size = int(self.image.shape[0] // self.side_zone_number)
        zone_avgs = []

        for i in range(self.side_zone_number):
            for j in range(self.side_zone_number):
                zone = self.image[i * side_size:i * side_size + side_size, j * side_size:j * side_size + side_size]
                subzones = self._get_horizontal_subzones(zone)
                zone_avgs.append(self._get_average_from_subzones(subzones))

        self.feature_vector = zone_avgs

    def get_feature_vector(self):
        return self.feature_vector

    # helper functions
    @staticmethod
    def _get_horizontal_subzones(zone):
        return [zone[i, :] for i in range(zone.shape[0])]

    @staticmethod
    def _get_average_from_subzones(subzones):
        """ Black pixels are considered as background and its value is 0.
        White pixels are considered as foreground (char) and its value is 1. """

        subzones_no = len(subzones)
        return np.sum([np.sum(subzones[i] == CHAR_PIXEL_VALUE) for i in range(subzones_no)]) / subzones_no


class EdgeMaps(FeatureDescriptor):
    """ This descriptor is responsible for creating feature vector from
    4 edge maps and preprocessed image. To get edge maps (horizontal,
    vertical, at angle of 45 and -45 degrees) Sobel operator is used.
    Every projection is split into a specified number of zones and
    in every zone percentage of char pixels is calculated. The percentage values
    are features for classifier.

    References:

    [1] R.M.O. Cruz, G.D.C. Cavalcanti, T.I. Ren, An Ensemble Classifier For Offline
    Cursive Character Recognition Using Multiple Feature Extraction Techniques """

    # angle of diagonal edge map
    ANGLE = 45

    def __init__(self, zone_number=16):
        super().__init__()
        self.side_zone_number = int(sqrt(zone_number))

    def preprocess(self, image):
        thresh = threshold_otsu(image)
        self.image = image > thresh

    def describe(self):
        """ This method assumes every projection has square form (N x N).
        Then in width and height number of zones is the same. In each zone
        of every projection and the preprocessed image is percentage
        of char pixels calculated.
        This percentages from every zone are features for classifier. """

        side_size = int(self.image.shape[0] // self.side_zone_number)
        skel_image = skeletonize(self.image)

        sv = sobel_v(skel_image)
        sh = sobel_h(skel_image)
        splus45 = np.sqrt((sv * sin(self.ANGLE)) ** 2 + (sh * cos(self.ANGLE)) ** 2)
        sminus45 = np.sqrt((sv * sin(-self.ANGLE)) ** 2 + (sh * cos(-self.ANGLE)) ** 2)

        projections = [self.image, self._binarize_projection(sv),
                       self._binarize_projection(sh),
                       self._binarize_projection(splus45),
                       self._binarize_projection(sminus45)]

        features = []

        for proj in projections:
            for i in range(self.side_zone_number):
                for j in range(self.side_zone_number):
                    zone = proj[i * side_size: i * side_size + side_size, j * side_size: j * side_size + side_size]
                    features.append(self._get_char_pixels_percentage(zone, side_size))

        self.feature_vector = features

    def get_feature_vector(self):
        return self.feature_vector

    @staticmethod
    def _get_char_pixels_percentage(zone, zone_side_size):
        return np.sum(zone == CHAR_PIXEL_VALUE) / (zone_side_size ** 2) * 100

    @staticmethod
    def _binarize_projection(projection):
        thresh = threshold_otsu(projection)
        return projection <= thresh


class ZoningChainCode(FeatureDescriptor):
    """ This class implements a descriptor that connects two other methods:
    Zoning and Freeman Chain Code. While Freeman Chain Code (FCC) is created,
    crossing of zone boundary is tracked. Then the FCC that was created in
    a specified zone, is assigned to the zone.

    Next FCC will be translated into simpler code according to SENSE2DIR dictionary
    - 8 senses will be converted into 4 directions.

    Next value of feature vector is result of analysis of chain of 4 directions in zone.
    1) no chain code in zone - 4 will be assigned as feature
    2) the most frequent value from chain code will be assigned as feature (dominant)
    3) if there are more than one most frequent number in chain code then te value will
    be chosen randomly from this dominant values.

    References:

    [1] Pseudo code of Freeman Chain Code: http://www.cs.unca.edu/~reiser/imaging/chaincode.html
    [2] Implementation example in Python: https://www.kaggle.com/mburger/freeman-chain-code-second-attempt """

    PAD_WIDTH = 5
    ITER_MAX = 1000
    NO_CHAIN_CODE = 'NO_CHAIN_CODE'

    # Map of directions
    DIR_MAP = [5, 6, 7,
               4,    0,
               3, 2, 1]

    # Moving in direction of columns
    X_MOVE = [-1, 0, 1,
              -1,    1,
              -1, 0, 1]

    # Moving in direction of rows
    Y_MOVE = [-1, -1, -1,
              0,       0,
              1,   1,  1]

    DIR2IDX = dict(zip(DIR_MAP, range(len(DIR_MAP))))

    # Generally, only in case of these constant SENSE and DIR (direction) refer
    # to vector features precisely. We have here 4 directions (specified by straights
    # without 'arrowhead') and 2x4 = 8 senses.
    # In other cases DIRECTION is used generally as these 8 senses in the code.
    SENSE2DIR = {0: 3, 1: 1, 5: 1, 4: 3, 2: 2, 6: 2, 3: 0, 7: 0, NO_CHAIN_CODE: 4}

    def __init__(self, zone_number=16):
        super().__init__()
        self.side_zone_number = int(sqrt(zone_number))
        self.zone_chain_codes = self._create_chain_code_containers()
        self.zone_boundary_map = None
        self.current_point = None

    def preprocess(self, image):
        thresh = threshold_otsu(image)
        self.image = image > thresh

        # add additional background pixels
        self.image = np.pad(self.image, self.PAD_WIDTH, mode='constant')
        self.zone_boundary_map = self._create_zone_boundary_map()

    def describe(self):
        start_px = self._find_start_point()

        # looking for 2nd point of char boundary
        last_direction = self._find_chain_code(self.DIR_MAP, start_px)

        count = 0
        # while the coordinates of the current pixel are not equal to those of the starting pixel
        while self.current_point != start_px:
            # figure direction to start search
            start_direction = (last_direction + 5) % 8
            # define order of directions which should be began searching from
            directions = []
            directions.extend(range(start_direction, 8))
            directions.extend(range(0, start_direction))

            last_direction = self._find_chain_code(directions, self.current_point)

            if count >= self.ITER_MAX:
                break
            count += 1

        # map FCC 8 directions on my 4 directions
        # determine dominant for each zone and append to feature vector
        self.feature_vector = self._chain_code2features()

    def get_feature_vector(self):
        return self.feature_vector

    def _create_zone_boundary_map(self):
        side_size = int((self.image.shape[0] - 2 * self.PAD_WIDTH) // self.side_zone_number)
        total_zone_number = self.side_zone_number ** 2
        keys = ['z' + str(i) for i in range(total_zone_number)]
        values = []

        for i in range(self.side_zone_number):
            for j in range(self.side_zone_number):
                lower_boundary = (i * side_size + self.PAD_WIDTH, j * side_size + self.PAD_WIDTH)
                upper_boundary = ((i + 1) * side_size + self.PAD_WIDTH, (j + 1) * side_size + self.PAD_WIDTH)
                values.append((lower_boundary, upper_boundary))

        zone_boundaries = dict(zip(keys, values))
        return zone_boundaries

    def _create_chain_code_containers(self):
        """ Each container corresponds to each zone. In the containers will be
        written chain codes of char being in zone. """

        zone_number = self.side_zone_number ** 2
        return dict(zip(['z' + str(i) for i in range(zone_number)], [list() for _ in range(zone_number)]))

    def _get_zone_based_on_coords(self, coords):
        y = coords[0]  # row
        x = coords[1]  # column

        for z in self.zone_boundary_map:
            zb = self.zone_boundary_map[z]
            if zb[0][0] <= y < zb[1][0] and zb[0][1] <= x < zb[1][1]:
                return z

    def _find_start_point(self):
        """ This method looks for starting point for Freeman chain code,
        i.e. first pixel of char boundary. """

        for i in range(self.PAD_WIDTH, self.image.shape[0] - self.PAD_WIDTH):
            for j in range(self.PAD_WIDTH, self.image.shape[1] - self.PAD_WIDTH):
                if self.image[i][j] == CHAR_PIXEL_VALUE:
                    return i, j

        raise Exception('Any foreground pixel not found!')

    def _find_chain_code(self, directions, prev_point):
        """ This function looks for next direction and write its code to chain code. """
        self.current_point = prev_point

        for direction in directions:
            idx = self.DIR2IDX[direction]
            new_point = (self.current_point[0] + self.Y_MOVE[idx], self.current_point[1] + self.X_MOVE[idx])

            if self.image[new_point] == CHAR_PIXEL_VALUE:
                z = self._get_zone_based_on_coords(new_point)
                self.zone_chain_codes[z].append(direction)
                self.current_point = new_point
                break

        return direction

    def _chain_code2features(self):
        features = []

        for zone_chain_code in self.zone_chain_codes.values():
            # convert senses to directions

            if not zone_chain_code:
                zone_dir_chain_code = [self.SENSE2DIR[self.NO_CHAIN_CODE]]
            else:
                zone_dir_chain_code = [self.SENSE2DIR[code] for code in zone_chain_code]

            if len(zone_dir_chain_code) == 1:
                features.append(zone_dir_chain_code[0])
                continue

            # find the most frequent or random from most frequents
            values, counts = np.unique(zone_dir_chain_code, return_counts=True)
            unique_map = dict(zip(values, counts))
            maxi = max(counts)

            unique_keys = []

            for key in unique_map:
                if unique_map[key] == maxi:
                    unique_keys.append(key)

            feature_from_zone = unique_keys[int(random() * len(unique_keys))]
            features.append(feature_from_zone)

        return features
