"""Evaluation helper - Bogen class."""
import os
import csv
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from scipy.linalg import norm

class Bogen(object):
    """Class of Bogen for every file to be evaluated."""

    def __init__(self, curr_file, ref_points, ref_boxes, isReference=False):
        """Initialization."""
        self.RefPosPath = "reference_positions.csv"
        self.image = Image.open(curr_file)
        self.curr_file = curr_file

        if isReference is False:
            self.ReferencePoints = ref_points
            self.CurrReferencePoints = self.findReferenceMasks()

            # positions of all boxes of the reference file
            self.ReferenceBoxes = ref_boxes
            self.CurrBoxes = ref_boxes

            # get transformation matrix
            self.transMatrix = self.getTransformationMatrix()

            # transform points
            self.PointsTransformed = False
            self.transformPositions()

            # save boxes to file
            self.extractBoxes()

        else:
            self.curr_file = curr_file
            self.ReferencePoints = self.findReferenceMasks()
            self.ReferenceBoxes = self.getRefPositions()

    def findReferenceMasks(self):
        """Find positions of the corners (see masks).

        The images are formated as 2479x2829.

        Parameters
        ----------
        image : str
            Path to image

        Returns
        -------
        lists of int
            Positions [x,y] of the reference points (corners).

        """
        # mask upper left
        mask_ul = Image.open("masks/ul.png")
        pos_ul = self.findPosition(mask_ul, x1=30, x2=110, y1=650,
                                   y2=730, xoff=52, yoff=52)

        # mask upper middle
        mask_um = Image.open("masks/ul.png")
        pos_um = self.findPosition(mask_um, x1=1070, x2=1150, y1=340,
                                   y2=430, xoff=52, yoff=52)

        # mask upper right
        mask_ur = Image.open("masks/ur.png")
        pos_ur = self.findPosition(mask_ur, x1=2300, x2=2400, y1=350,
                                   y2=430, xoff=54, yoff=59)

        # mask lower left
        mask_ll = Image.open("masks/ll.png")
        pos_ll = self.findPosition(mask_ll, x1=30, x2=110, y1=2720,
                                   y2=2791, xoff=52, yoff=36)

        # mask lower right
        mask_lr = Image.open("masks/lr.png")
        pos_lr = self.findPosition(mask_lr, x1=2300, x2=2400, y1=2710,
                                   y2=2789, xoff=51, yoff=54)

        return pos_ul, pos_um, pos_ur, pos_ll, pos_lr

    def findPosition(self, mask, x1, x2, y1, y2, xoff, yoff):
        """Find position of a mask in a specific window."""
        mask_px = mask.load()
        pos = [0, 0]
        mse_min = 999999

        # scan through predefined area of bogen picture
        for x in range(x1, x2):
            for y in range(y1, y2):
                # extract rectangular on which corner is searched
                px = self.image.crop([x, y, x+xoff, y+yoff]).load()

                # calculate mean squared error between the two cropped images
                mse_curr = self.getMSEBetweenTwoImages(mask_px, px)

                # find the minimum mse
                if mse_curr < mse_min:
                    mse_min = mse_curr
                    pos = [x, y]

        # print(mse_min, pos)
        return pos

    def getMSEBetweenTwoImages(self, im1, im2, x_max=36, y_max=36):
        """Calculate mean squared error (MSE) between two images pixel-wise.

        Parameters
        ----------
        im1, im2 : images
            Images to be compared with
        x_max, y_max : int, optional
            Maximum values of x and y

        Returns
        -------
        float
            Mean squared error

        """
        mse = 0
        for i in range(y_max):
            for j in range(x_max):
                # compare pixel wise, ie "value" of each pixel
                mse += (im1[i, j] - im2[i, j])**2
        return mse/(y_max*x_max)

    def getTransformationMatrix(self):
        """Get transformation matrix between reference and new file.

        Parameters
        ----------
        refPoints : list of lists
            Reference points of reference file
        newPoints : list of lists
            Reference points of current file

        Returns
        -------
        list of lists
            Matrix A and vector b of the linear transformation

        """
        def errFunc(x):
            A = x[:4].reshape(2, 2)
            b = x[4:]
            return sum([(norm(np.dot(A, self.ReferencePoints[i]) + b -
                              self.CurrReferencePoints[i]))**2
                        for i in range(len(self.ReferencePoints))])

        x0 = [1., 0., 0., 1., 0., 0.]
        res = minimize(errFunc, x0, method="nelder-mead",
                       options={'maxiter': 10000000})   # 'xatol': 1e-12,
        # print("transMatrix = ", res.x)
        return res.x

    def transformPositions(self):
        """Get new box positions of file.

        Parameters
        ----------
        tmatrix : list of float
            Matrix A and vector b of the linear transformation
        path : str
            Path to file
        boegen_folder : str
            Path to folder of boegen

        Returns
        -------
        refpos :  : dict
            New dictionary of all reference positions

        """
        if self.PointsTransformed is True:
            print("Points already transformed.")
            return

        # transform point for point
        for i in range(len(self.ReferenceBoxes["x0"])):
            pointUL = [self.ReferenceBoxes["x0"][i],
                       self.ReferenceBoxes["y0"][i]]
            newPointUL = self.transformPoint(pointUL)

            pointLR = [self.ReferenceBoxes["x1"][i],
                       self.ReferenceBoxes["y1"][i]]
            newPointLR = self.transformPoint(pointLR)

            self.CurrBoxes["x0"][i] = newPointUL[0]
            self.CurrBoxes["y0"][i] = newPointUL[1]
            self.CurrBoxes["x1"][i] = newPointLR[0]
            self.CurrBoxes["y1"][i] = newPointLR[1]

        self.PointsTransformed = True

    def transformPoint(self, point):
        """Transform point with transformation matrix.

        Parameters
        ----------
        point : list of int
            2D point to be transformed

        Returns
        -------
        newpoint : list of float
            Transformed 2D point

        """
        A = self.transMatrix[:4].reshape(2, 2)
        b = self.transMatrix[4:]
        newpoint = np.dot(A, point) + b
        return newpoint

    def extractBoxes(self):
        """Extract boxes from current file.

        Parameters
        ----------
        refpos : dict
            Dictionary of all reference positions
        path : str
            Path to file

        Returns
        -------
        images : png
            Boxes as images

        """
        # loop for rectangles
        for i in range(len(self.CurrBoxes["x0"]) - 4):

            # get corner points
            corners = [self.CurrBoxes["x0"][i]-10, self.CurrBoxes["y0"][i]-10,
                       self.CurrBoxes["x1"][i]+10, self.CurrBoxes["y1"][i]+10]

            # cut box out of image
            box = self.image.crop(corners)

            # resize to 40x40 image
            box = box.resize((40, 40))

            # iterative folder
            # necessary as glob.glob has a kind of random sorting
            path = self.curr_file
            folder = path[path.find("/Bogen")+1:path.find(".")]

            # path and folder
            directory = "boxes/" + folder

            # check if folder exists
            if not os.path.exists(directory):
                os.makedirs(directory)

            # iterative name
            name = "box" + str(i)

            # save as image
            box.save(directory + "/" + name + ".png")

    def getRefPositions(self):
        """Get reference positions from CSV file.

        Returns
        -------
        posdict : dict
            Dictionary containing the positions of the reference file.

        """
        posdict = {}

        with open(self.RefPosPath) as posfile:
            reader = csv.DictReader(posfile, delimiter="\t",
                                    skipinitialspace=True)
            for i, row in enumerate(reader):
                for c, v in row.items():
                    if c == "question" or c == "box":
                        posdict.setdefault(c, []).append(v)
                    else:
                        posdict.setdefault(c, []).append(int(v))
        # print(posdict)
        return posdict

    def drawFile(self):
        """Draw file with all rectangles.

        Parameters
        ----------
        refpos : dict
            Dictionary of all reference positions
        path : str
            Path to file

        """
        draw = ImageDraw.Draw(self.image)
        for i in range(len(self.CurrReferencePoints["x0"])):
            box = [self.CurrReferencePoints["x0"][i],
                   self.CurrReferencePoints["y0"][i],
                   self.CurrReferencePoints["x1"][i],
                   self.CurrReferencePoints["y1"][i]]

            draw.rectangle(box, fill="Black")
        # im.show()
