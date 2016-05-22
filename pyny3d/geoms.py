# -*- coding: utf-8 -*-
import numpy as np

class root(object):
    """
    Lowest geometry class in hierarchy. Actually do nothig but store
    two general methods for the real classes:
    
        * :func:`plot`
        * :func:`get_centroid`
        * :func:`copy`
        * :func:`save`
        * :func:`restore`
    
    Other Global methods (but individually defined in each class) are:
    
        * get_seed
        * seed2pyny
        * get_domain
        
    """
    def __init__(self):
        self.backup = None
    
    def plot(self, color='default', ret=False, ax=None):
        """
        Generates a basic 3D visualization.
  
        :param color: Polygons color.
        :type color: matplotlib color, 'default' or 't' (transparent)
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :param ax: If a matplotlib axes given, this method will 
            represent the plot on top of this axes. This is used to
            represent multiple plots from multiple geometries, 
            overlapping them recursively.
        :type ax: mplot3d.Axes3D, None
        :returns: None, axes
        :rtype: mplot3d.Axes3D, bool
        """
        import matplotlib.pylab as plt
        import mpl_toolkits.mplot3d as mplot3d
        
        # Bypass a plot
        if color == False:
            if ax is None: ax = mplot3d.Axes3D(fig=plt.figure())
            return ax
        
        # Clone and extract the information from the object
        obj = self.__class__(**self.get_seed())
        plotable3d = obj.get_plotable3d()
            
        # Domain
        domain = obj.get_domain()
        bound = np.max(domain[1]-domain[0])
        centroid = obj.get_centroid()
        pos = np.vstack((centroid-bound/2, centroid+bound/2))
        
        # Cascade plot?
        if ax is None: # Non cascade
            ax = mplot3d.Axes3D(fig=plt.figure())
        else:
            old_pos = np.array([ax.get_xbound(),
                                ax.get_ybound(),
                                ax.get_zbound()]).T
            pos = np.dstack((pos, old_pos))
            pos = np.array([np.min(pos[0, :, :], axis=1),
                            np.max(pos[1, :, :], axis=1)])
            
        # Plot
        if color == 'default': color = 't'
        if color == 't': color = (0,0,0,0)
            
        for polygon in plotable3d:
            polygon.set_facecolor(color)
            polygon.set_edgecolor('k')
            ax.add_collection3d(polygon)
        
        # Axis limits
        ax.set_xlim3d(left=pos[0,0], right=pos[1,0])
        ax.set_ylim3d(bottom=pos[0,1], top=pos[1,1])
        ax.set_zlim3d(bottom=pos[0,2], top=pos[1,2])
        
        if ret: return ax
            
    def center_plot(self, ax):
        """
        Centers and keep the aspect ratio in a 3D representation.
        
        Created to help higher classes to manage cascade representation
        of multiple lower objects.
        
        :param ax: Axes to apply the method.
        :type ax: mplot3d.Axes3D
        :returns: None
        """
        
        # Domain
        domain = self.get_domain()
        bound = np.max(domain[1]-domain[0])
        centroid = self.get_centroid()
        pos = np.vstack((centroid-bound/2, centroid+bound/2))

        # Axis limits
        ax.set_xlim3d(left=pos[0,0], right=pos[1,0])
        ax.set_ylim3d(bottom=pos[0,1], top=pos[1,1])
        ax.set_zlim3d(bottom=pos[0,2], top=pos[1,2])

    def get_centroid(self):
        """
        The centroid is considered the center point of the circunscribed
        paralellepiped, not the mass center.
        
        :returns: (x, y, z) coordinates of the centroid of the object.
        :rtype: ndarray
        """
        return self.get_domain().mean(axis=0)
    
    def copy(self):
        """
        :returns: A deepcopy the entire instance.
        :rtype: ``pyny3d`` object
        
        .. seealso:: :func:`save`, :func:`restore`
        """
        import copy
        return self.__class__(**copy.deepcopy(self.get_seed()))
        
    def save(self):
        """
        Saves a deepcopy of the current state the instance. 
        ``.restore()`` method will return this copy.

        :returns: None
        
        .. seealso:: :func:`restore`, :func:`copy`
        """
        self.backup = self.copy()
        
    def restore(self):
        """
        Load a previous saved state of the current object. ``.save()``
        method can be used any time to save the current state of an 
        object.
        
        :returns: Last saved version of this object.
        :rtype: ``pyny3d`` object

        .. seealso:: :func:`save`, :func:`copy`
        """
        if self.backup is not None:
            return self.backup
        else:
            raise ValueError('No backup previously saved.')

        
class Polygon(root):
    """
    The most basic geometry class. It generates and stores all the 
    information relative to a 3D polygon.
    
    Instances of this class work as iterable object. When indexed, 
    returns the points which conform it.
    
    :param points: Sorted points which form the polygon (xyz or xy). 
        Do not repeat the first point at the end.
    :type points: ndarray *shape=(N, 2 or 3)*
    :param check_convexity: If True, an error will be raised for 
        concave Polygons. It is a requirement of the code that the
        polygons have to be convex.
    :type check_convexity: bool
    :returns: None
  
    .. note:: This object can be locked (``.lock()`` method) in order to 
        precompute information for faster further computations.
    """
    verify = True
    def __init__(self, points, make_ccw=True, **kwargs):
        
        # Input errors
        if type(points) != np.ndarray:
            raise ValueError('pyny3d.Polygon needs a ndarray as input')
                
        # Adapt 2D/3D
        if points.shape[1] == 2:
            from pyny3d.utils import arange_col
            points = np.hstack((points, arange_col(points.shape[0])*0))
        elif points.shape[1] != 3:
            raise ValueError('pyny3d.Polygon needs 2 or 3 coords '+\
                             '(columns) at least')
        if make_ccw and Polygon.verify: points = Polygon.make_ccw(points)

        # Basic processing
        self.points = points
        
        # Optional processing
        self.path = None
        self.parametric = None
        self.shapely = None
        
        # Parameters
        self.locked = False
        self.domain = None
        self.area = None
                
    def __iter__(self): return iter(self.points)
    def __getitem__(self, key): return self.points[key]
        
    def lock(self):
        """
        Precomputes some parameters to run faster specific methods like
        Surface.classify.

        Stores ``self.domain`` and ``self.path``, both very used in
        the shadows simulation, in order to avoid later unnecessary 
        calculations and verifications.
        
        :returns: None
        
        .. warning:: Unnecessary locks can slowdown your code.
        """
        if not self.locked:
            self.path = self.get_path()
            self.domain = self.get_domain()
            self.locked = True

    def seed2pyny(self, seed):
        """
        Re-initialize an object with a seed.
        
        :returns: A new ``pyny.Polygon``
        :rtype: ``pyny.Polygon``
        """
        # import geoms as pyny
        return Polygon(**seed)

    @staticmethod
    def is_convex(points):
        """
        Static method. Returns True if the polygon is convex regardless 
        of whether its vertices follow a clockwise or a 
        counter-clockwise order. This is a requirement for the rest of 
        the program.
        
        :param points: Points intented to form a polygon.
        :type points: ndarray with points xyz in rows
        :returns: Whether a polygon is convex or not.
        :rtype: bool
        
        .. note:: Despite the code works for ccw polygons, in order to 
            avoid possible bugs it is always recommended to use ccw 
            rather than cw.
            
        .. warning:: This method do not check the order of the points.
        """
        # Verification based on the cross product
        n_points = points.shape[0]
        i=-1
        u = points[i] - points[i-1]
        v = points[i+1] - points[i]
        last = np.sign(np.round(np.cross(u, v)))
        while i < n_points-1:
            u = points[i] - points[i-1]
            v = points[i+1] - points[i]
            s = np.sign(np.round(np.cross(u, v)))
            if abs((s - last).max()) > 1:
                return False
            last = s
            i += 2
        return True

    @staticmethod
    def make_ccw(points):
        """
        Static method. Returns a counterclock wise ordered sequence of 
        points. If there are any repeated point, the method will raise 
        an error.
        
        Due to the 3D character of the package, the order or the points
        will be tried following this order:
		
            1. z=0 pprojection
            2. x=0 pprojection
            3. y=0 pprojection

        :param points: Points to form a polygon (xyz or xy)
        :type points: ndarray with points (xyz or xy) in rows
        :returns: ccw version of the points.
        :rtype: ndarray (shape=(N, 2 or 3))
        """
        from scipy.spatial import ConvexHull
        from pyny3d.utils import sort_numpy
        
        # Repeated points
        points_aux = sort_numpy(points)
        check = np.sum(np.abs(np.diff(points_aux, axis=0)), axis=1)
        if check.min() == 0: raise ValueError('Repeated point: \n'+str(points))
        
        # Convexity
        hull = None
        for cols in [(0, 1), (1, 2), (0, 2)]:
            try:
                hull = ConvexHull(points[:, cols])
            except:
                pass
            if hull is not None: return points[hull.vertices]
        if hull is None: raise ValueError('Wrong polygon: \n'+str(points))

    def to_2d(self):
        """
        Generates the real 2D polygon of the 3D polygon. This method
		performs a change of reference system obtaining the same polygon
		but with the new z=0 plane containing the polygon.
        
        This library mostly uses the z=0 projection to perform 
        operations with the polygons. For this reason, if real 2D
        planar operations are required (like calculate real area) the 
        best way is to create a new ``pyny.Polygon`` with this method.
        
        :returns: Planar orthogonal view of the polygon.
        :rtype: ``pyny.Polygon``
        """
        # New reference system
        a = self[1]-self[0]
        a = a/np.linalg.norm(a) # arbitrary first axis
        n = np.cross(a, self[-1]-self[0])
        n = n/np.linalg.norm(n) # normal axis
        b = -np.cross(a, n) # Orthogonal to the others
        
		# Reference system change
        R_inv = np.linalg.inv(np.array([a, b, n])).T
        real = np.dot(R_inv, self.points.T).T
        real[np.isclose(real, 0)] = 0
        
        return Polygon(real[:, :2])

    def contains(self, points, edge=True):
        """
        Point-in-Polygon algorithm for multiple points for the z=0 
        projection of the ``pyny.Polygon``.
        
        :param points: Set of points to evaluate.
        :type points: ndarray with points (xyz or xy) in rows
        :param edge: If True, consider the points in the Polygon's edge
            as inside the Polygon.
        :type edge: bool
        :returns: Whether each point is inside the polygon or not (in
            z=0 projection).
        :rtype: ndarray (dtype=bool)
        """
        radius = 1e-10 if edge else -1e-10
        return self.get_path().contains_points(points[:, :2],
                                               radius=radius)

    def get_parametric(self, check=True, tolerance=0.001):
        """
        Calculates the parametric equation of the plane that contains 
        the polygon. The output has the form np.array([a, b, c, d]) 
        for:

        .. math::
            a*x + b*y + c*z + d = 0
        
        :param check: Checks whether the points are actually
            in the same plane with certain *tolerance*.
        :type check: bool
        :param tolerance: Tolerance to check whether the points belong
            to the same plane.
        :type tolerance: float
        
        .. note:: This method automatically stores the solution in order
            to do not repeat calculations if the user needs to call it 
            more than once.
        """
        if self.parametric is None: 
            
            # Plane calculation
            a, b, c = np.cross(self.points[2,:]-self.points[0,:],
                               self.points[1,:]-self.points[0,:])
            d = -np.dot(np.array([a, b, c]), self.points[2, :])
            self.parametric = np.array([a, b, c, d])
                
            # Point belonging verification
            if check:
                if self.points.shape[0] > 3:
                    if np.min(np.abs(self.points[3:,0]*a+
                                     self.points[3:,1]*b+
                                     self.points[3:,2]*c+
                                     d)) > tolerance:
                        raise ValueError('Polygon not plane: \n'+\
                                         str(self.points))
        return self.parametric
        
    def get_path(self):
        """
        :returns: matplotlib.path.Path object for the z=0 projection of 
            this polygon.
        """
        if self.path == None:
            from matplotlib import path
            return path.Path(self.points[:, :2]) # z=0 projection!
        return self.path
        
    def get_shapely(self):
        """
        :returns: shapely.Polygon object of the z=0 projection of 
            this polygon.
        """
        if self.shapely == None:
            from shapely.geometry import Polygon as shPolygon
            self.shapely = shPolygon(self.points[:, :2]) # z=0 projection!
        return self.shapely
            
    def get_domain(self):
        """
        :returns: opposite vertices of the bounding prism for this 
            object.
        :rtype: ndarray([min], [max])
        """
        if self.domain is None:
            return np.array([self.points.min(axis=0), 
                             self.points.max(axis=0)])
        return self.domain

    def get_area(self):
        """
        :returns: The area of the polygon.
        """
        if self.area is None:
            self.area = self.to_2d().get_shapely().area
        return self.area

    def get_height(self, points, only_in = True, edge=True, full=False):
        """
        Given a set of points, it computes the z value for the 
		parametric equation of the plane where the polygon belongs.
        
        Only the two first columns of the points will be taken into 
        account as x and y.
        
        By default, the points outside the object will have a NaN value 
        in the z column. If the inputed points has a third column the z 
        values outside the Surface's domain will remain unchanged, the 
        rest will be replaced.
        
        :param points: Coordinates of the points to calculate.
        :type points: ndarray shape=(N, 2 or 3)
        :param only_in: If True, computes only the points which are 
            inside of the Polygon.
        :type only_in: bool
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :param full: If true, the return will have three columns 
            (x, y, z) instead of one (z).
        :type full: bool
        :returns: (z) or (x, y, z)
        :rtype: ndarray shape=(N, 1 or 3)
        """
        
        p = self.get_parametric()
        z = (-p[0]*points[:, 0]-p[1]*points[:, 1]-p[3])/p[2]
        
        if only_in:
            pip = self.contains(points, edge=edge)
            z[pip == False] *= np.nan
            
        if full:
            z = np.hstack((points[:, :2], 
                           np.reshape(z, (points.shape[0], 1))))
            if points.shape[1] == 3: # Restore original z
                z[pip == False] = points[pip == False]
        return z
            
    def get_seed(self):
        """
        Collects the required information to generate a data estructure 
        that can be used to recreate exactly the same geometry object
        via *\*\*kwargs*.
        
        :returns: Object's sufficient info to initialize it.
        :rtype: dict
        """
        return {'points': self.points}

    def get_plotable3d(self):
        """
        :returns: matplotlib Poly3DCollection
        :rtype: mpl_toolkits.mplot3d
        """
        import mpl_toolkits.mplot3d as mplot3d
        return [mplot3d.art3d.Poly3DCollection([self.points])]
        
    def pip(self, points, sorted_col=0, radius=0):
        """
        Point-in-Polygon for the z=0 projection. This function enhances
        the performance of ``Polygon.contains()`` by verifying only the 
        points which are inside the bounding box of the polygon. To do 
        it fast, it needs the points array to be already sorted by one
        column.
        
        :param points: list of *(x, y, z) or (x, y)* coordinates of the
            points to check. (The z value will not be taken into 
            account).
        :type points: ndarray (shape=(N, 2 or 3))
        :param sorted_col: Index of the sorted column (0 or 1).
        :type sorted_col: int
        :param radius: Enlarge Polygons domain by a specified quantity.
        :type radius: float
        :returns: Which points are inside the polygon.
        :rtype: ndarray (dtpye=bool)
        
        .. warning:: By default pip considers that the set of points is
            currently sorted by the first column.
        .. warning:: This method only works if the polygon has been 
            locked (:func:`lock`).
        """
        xy = points[:, :2]
        n_points = xy.shape[0]
        index = np.arange(n_points, dtype = int)
        b = self.domain
        b[0] = b[0] - radius
        b[1] = b[1] + radius
        
        # Slicing the sorted column
        k = np.searchsorted(xy[:, sorted_col],
                            (b[0, sorted_col], b[1, sorted_col]+1e-10))
        xy = xy[k[0]:k[1]]
        index = index[k[0]:k[1]]
        
        # solution
        k = index[self.path.contains_points(xy, radius=radius)]
        sol = np.zeros(n_points, dtype=bool)
        sol[k] = True
                
        return sol
        
    def plot2d(self, color='default', alpha=1, ret=True):
        """
        Generates a 2D plot for the z=0 Polygon projection.
        
        :param color: Polygon color.
        :type color: matplotlib color
        :param alpha: Opacity.
        :type alpha: float
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :returns: None, axes
        :rtype: None, matplotlib axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        path = self.get_path()
        domain = self.get_domain()[:, :2]

        if color is 'default': color = 'b'

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.add_patch(patches.PathPatch(path, facecolor=color, lw=1, 
                                       edgecolor='k', alpha=alpha))
        ax.set_xlim(domain[0,0],domain[1,0])
        ax.set_ylim(domain[0,1], domain[1,1])
        if ret: return ax

    def move(self, d_xyz):
        """
        Translate the Polygons in x, y and z coordinates.
        
        :param d_xyz: displacement in x, y(, and z).
        :type d_xyz: tuple (len=2 or 3)
        :returns: ``pyny.Polygon``
        """
        space = Space(Place(Surface(self)))
        return space.move(d_xyz, inplace=False)[0].surface[0]

    def rotate(self, angle, direction='z', axis=None):
        """
        Returns a new Polygon which is the same but rotated about a 
        given axis.
        
        If the axis given is ``None``, the rotation will be computed
        about the Surface's centroid.
        
        :param angle: Rotation angle (in radians)
        :type angle: float
        :param direction: Axis direction ('x', 'y' or 'z')
        :type direction: str
        :param axis: Point in z=0 to perform as rotation axis
        :type axis: tuple (len=2 or 3) or None
        :returns: ``pyny.Polygon``
        """
        space = Space(Place(Surface(self)))
        return space.rotate(angle, direction, axis)[0].surface[0]
        
    def mirror(self, axes='x'):
        """
        Generates a symmetry of the Polygon respect global axes.
        
        :param axes: 'x', 'y', 'z', 'xy', 'xz', 'yz'...
        :type axes: str
        :returns: ``pyny.Polygon``
        """
        space = Space(Place(Surface(self)))
        return space.mirror(axes, inplace=False)[0].surface[0]

    def matrix(self, x=(0, 0), y=(0, 0) , z=(0, 0)):
        """
        Copy the ``pyny.Polygon`` along a 3D matrix given by the 
        three tuples x, y, z:        

        :param x: Number of copies and distance between them in this
            direction.
        :type x: tuple (len=2)
        :returns: list of ``pyny.Polygons``
        """
        space = Space(Place(Surface(self)))
        space = space.matrix(x, y, z, inplace=False)
        return [place.surface[0] for place in space]

        
class Surface(root):
    """
    This class groups contiguous polygons (coplanars or not). These
    polygons cannot overlap each other on the z=0 projection\*.
    
    This object is a composition of polygons and holes. The polygons can
    be used to "hold up" other objects (points, other polygons...) and
    to compute shadows. The holes exist only to prevent the program 
    to place objects on them. The shadows computation do not take care
    of the holes\*\*, instead, they can be emulated by a collection of 
    polygons.
    
    Instances of this class work as iterable object. When indexed, 
    returns the ``pyny.Polygons`` which conform it.

    :param polygons: Polygons to be set as Surface. This is the only
        necessary input to create a Surface.
    :type polygons: list of ndarray, list of ``pyny.Polygon``
    :param holes: Polygons to be set as holes of the Surface.
    :type holes: list of ndarray, list of ``pyny.Polygon``
    :param make_ccw: If True, points will be sorted ccw for each 
        polygon.
    :type make_ccw: bool
    :param melt: If True, the :func:`melt` method will be launched at
        initialization.
    :type melt: bool
    :param check_contiguity: If True, :func:`contiguous` will be 
        launched at initialization.
    :type check_contiguity: bool
    
    :returns: None
    
    .. note:: \* For models with planes stacked in column, use
        the Place class to distinct them. For example, a three-storey 
        building structure can be modeled by using one ``pyny.Place`` 
        for storey where the floor is a Surface and the columns are 
        Polyhedra.
    .. note:: \*\* In the future versions of this library it will 
        simulate shadows through the holes.
    """
    def __init__(self, polygons, holes=[], make_ccw=True, 
                 melt=False, check_contiguity=False, **kwargs):
        # Always works with lists
        if type(polygons) != list: polygons = [polygons]
        if type(holes) != list: holes = [holes]
        
        # Creating the object
        ## Polygons
        if type(polygons[0]) == np.ndarray:
            self.polygons = [Polygon(polygon, make_ccw)
                             for polygon in polygons]
        elif type(polygons[0]) == Polygon:
            self.polygons = polygons
        else:
            raise ValueError('pyny3d.Surface needs a ndarray or '+\
            'pyny3d.Polygons as input')
        
        ### Check contiguity
        if check_contiguity:
            if not Surface.contiguous(self.polygons):
                raise ValueError('Non-contiguous polygons in the Surface')
        
        ## Holes
        if len(holes) > 0:
            if type(holes[0]) == np.ndarray:
                self.holes = [Polygon(hole, make_ccw)
                              for hole in holes]
            elif type(holes[0]) == Polygon:
                self.holes = holes
        else:
            self.holes = []
        
        if melt: self.melt()
            
    def __iter__(self): return iter(self.polygons)    
    def __getitem__(self, key): return self.polygons[key]
    
    def lock(self):
        """
        Lock the Polygons in the Surface to run faster specific methods 
		like Surface.classify.
        
        :returns: None
        """
        for polygon in self.polygons: polygon.lock()

    def seed2pyny(self, seed):
        """
        Re-initialize an object with a seed.
        
        :returns: A new ``pyny.Surface``
        :rtype: ``pyny.Surface``
        """
        # import geoms as pyny
        return Surface(**seed)
            
    def classify(self, points, edge=True, col=1, already_sorted=False):
        """
        Calculates the belonging relationship between the polygons
        in the Surface and a set of points.
        
        This function enhances the performance of ``Polygon.contains()``
		when used with multiple non-overlapping polygons (stored in a 
		Surface) by verifying only the points which are inside the z=0 
		bounding box of each polygon. To do it fast, it sorts the points
		and then apply ``Polygon.pip()`` for each Polygon.
        
        :param points: list of (x, y, z) or (x, y) coordinates of the
            points to check. (The z value will not be taken into 
            account).
        :type points: ndarray (shape=(N, 2 or 3))
        :param edge: If True, consider the points in a Polygon's edge
            inside a Polygon.
        :type edge: bool
        :param col: Column to sort or already sorted.
        :type col: int
        :param already_sorted: If True, the method will consider that
            the *points* are already sorted by the column *col*.
        :type already_sorted: bool
        :returns: Index of the Polygon to which each point belongs. 
            -1 if outside the Surface.
        :rtype: ndarray (dtpye=int)
        """
        xy = points[:, :2].copy()
        radius = 1e-10 if edge else 0
        n_points = xy.shape[0]
        self.lock() # Precomputes polygons information
        
        # Sorting the points
        if not already_sorted:
            from pyny3d.utils import sort_numpy
            # Sorting by the y column can be faster if set_mesh has been
            # used to generate the points.        
            points, order_back = sort_numpy(xy, col, order_back=True)
        
        # Computing PiP
        sol = np.ones(n_points, dtype=int)*-1
        for i, polygon in enumerate(self):
            pip = polygon.pip(points, sorted_col=col, radius=radius)
            sol[pip] = i
        
        # Return
        if not already_sorted:
            return sol[order_back]
        else:
            return sol

    def intersect_with(self, polygon):
        """
        Calculates the intersection between the polygons in this surface
        and other polygon, in the z=0 projection.
        
        This method rely on the ``shapely.Polygon.intersects()`` method.
        The way this method is used is intersecting this polygon 
        recursively with all identified polygons which overlaps with it
        in the z=0 projection.
        
        :param polygon: Polygon to intersect with the Surface.
        :type polygon: pyny.Polygon
        :returns: Multiple polygons product of the intersections.
        :rtype: dict of ndarrays (keys are the number of the polygon
            inside the surface)
        """
        
        intersections = {}
        for i, poly in enumerate(self):
            if polygon.get_shapely().intersects(poly.get_shapely()):
                inter = polygon.get_shapely().intersection(poly.get_shapely())
                intersections[i] = np.array(list(inter.exterior.coords))[:-1]
        return intersections
            
    def get_plotable3d(self):
        """
        :returns: matplotlib Poly3DCollection
        :rtype: list of mpl_toolkits.mplot3d
        """
        return [polygon.get_plotable3d()[0] for polygon in self]
       
    def get_domain(self):
        """
        :returns: opposite vertices of the bounding prism for this 
            object in the form of ndarray([min], [max])
        
        .. note:: This method automatically stores the solution in order
            to do not repeat calculations if the user needs to call it 
            more than once.
        """
        points = ([poly.points for poly in self]+
                  [holes.points for holes in self.holes])
        points = np.concatenate(points, axis=0)
        return np.array([points.min(axis=0), points.max(axis=0)])

    def get_seed(self):
        """
        Collects the required information to generate a data estructure 
        that can be used to recreate exactly the same geometry object
        via *\*\*kwargs*.
        
        :returns: Object's sufficient info to initialize it.
        :rtype: dict
        """
        return {'polygons': [poly.points for poly in self],
                'holes': [hole.points for hole in self.holes]}
            
            
    def get_height(self, points, edge=True):
        """
        Given a set of points, computes the z value for the parametric
        equation of the Polygons in the Surface.
        
        This method computes recursively the ``Polygon.get_height()``
        method for all the Polygons in the Surface, obtaining the z 
        value for the points according to the local Polygon they belong.

        The points outside the object will have a NaN value in the
        z column. If the inputed points has a third column the z values
        outside the Surface's domain will remain unchanged, the rest
        will be replaced.

        :param points: list of coordinates of the points to calculate.
        :type points: ndarray (shape=(N, 2 or 3))
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :returns: (x, y, z) arrays
        :rtype: ndarray (shape=(N, 3))
        """
        for poly in self:
            points = poly.get_height(points, edge=edge, full=True)
        for hole in self.holes:
            pip = hole.contains(points, edge=True)
            points[pip, 2] = np.nan
        return points
        
    def add_holes(self, holes_list, make_ccw=True):
        """
        Add holes to the holes list.
        
        :param holes_list: Polygons that will be treated as holes.
        :type holes_list: list or pyny.Polygon
        :param make_ccw: If True, points will be sorted ccw.
        :type make_ccw: bool
        :returns: None
        
        .. note:: The holes can be anywhere, not necesarily on the 
            surface.
        """
        if type(holes_list) != list: holes_list = [holes_list]
        self.holes += [Polygon(hole, make_ccw) for hole in holes_list]
                           
    def melt(self, plot=False):
        """
        Find and merge groups of polygons in the surface that meet the 
        following criteria:
        
            * Are coplanars.
            * Are contiguous.
            * The result is convex.
            
        This method is very useful at reducing the number the items and,
        therefore, the shadowing time computing. Before override this
        instance, it is saved and can be restored with ``.restore()``
        
        :param plot: If True, generates the before and after 
            visualizations for the surface. Use it to check the results.
        :type plot: bool
        :returns: None
        
        .. warning:: This method do not check if the merged polygons are 
            actually convex. The convex hull of the union is directly 
            calculated. For this reason, it is very important to visualy
            check the solution.
        """
        from pyny3d.utils import bool2index
        from scipy.spatial import ConvexHull
        
        # First, coplanarity
        ## Normalize parametric equations
        para = [poly.get_parametric() for poly in self]
        para = np.array([p/np.linalg.norm(p) for p in para])
        n = para.shape[0]
        ## Coincidences
        cop = []
        for i, plane in enumerate(para[:-1]):
            indexes = np.zeros((n-i-1, 4))
            for c in range(4):
                indexes[:, c] = np.isclose(para[i+1:, c], plane[c])
            pos = bool2index(indexes.sum(axis=1)==4)+i+1
            if pos.shape[0] > 0:
                cop.append(np.hstack((i, pos)))
                para[pos, :] = np.nan
        
        # Second, contiguity
        substituted = []
        cop_cont = []
        for i, group in enumerate(cop):
            polygons = [self[i] for i in group]
            if Surface.contiguous(polygons):
                cop_cont.append(polygons)
                substituted.append(group)
                        
        if len(substituted) != 0:
            self.save()
            if plot: self.plot()
            substituted = sum(substituted)
                    
            # Hull        
            merged = []
            for polygons in cop_cont:
                points = np.concatenate([polygon.points 
                                         for polygon in polygons])
                hull = ConvexHull(points[:, :2])
                merged.append(Polygon(points[hull.vertices]))
    
            # Final substitution
            new_surface = [self[i] for i in range(len(self.polygons))
                                   if i not in substituted]
            new_surface += merged
            self.polygons = new_surface
            self.sorted_areas = None
            
            if plot: self.plot()

    def get_area(self):
        """
        :returns: The area of the surface.
        
        .. warning:: The area is computed as the sum of the areas of all
            the polygons minus the sum of the areas of all the holes.
        """
        polys = sum([polygon.get_area() for polygon in self])
        holes = sum([hole.get_area() for hole in self.holes])
        return polys-holes

    @staticmethod        
    def contiguous(polygons):
        """
        Static method. Check whether a set of convex polygons are all 
		contiguous. Two polygons are considered contiguous if they 
		share, at least, one side (two vertices).
        
        This is not a complete verification, it is a very simplified
        one. For a given set of polygons this method will verify that 
        the number of common vertices among them equals or exceeds the
        minimum number of common vertices possible.
        
        This little algorithm will not declare a contiguous set of 
        polygons as non-contiguous, but it can fail in the reverse for 
        certain geometries where polygons have several common vertices
        among them.
        
        :param polygons: List of polygons.
        :type polygons: list of ``pyny.Polygon``
        :return: Whether tey are contiguous.
        :rtype: bool
        """
        from pyny3d.utils import sort_numpy
        
        n = len(polygons)
        points = sort_numpy(np.concatenate([polygon.points 
                                            for polygon in polygons]))
        diff = np.sum(np.diff(points, axis=0), axis=1)
        if sum(np.isclose(diff, 0)) < n*2-2:
            return False
        else:
            return True

    def plot2d(self, c_poly='default', alpha=1, cmap='default', ret=False, 
               title=' ', colorbar=False, cbar_label=''):
        """
        Generates a 2D plot for the z=0 Surface projection.
        
        :param c_poly: Polygons color.
        :type c_poly: matplotlib color
        :param alpha: Opacity.
        :type alpha: float
        :param cmap: colormap
        :type cmap: matplotlib.cm
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :param title: Figure title.
        :type title: str
        :param colorbar: If True, inserts a colorbar in the figure.
        :type colorbar: bool
        :param cbar_label: Colorbar right label.
        :type cbar_label: str
        :returns: None, axes
        :rtype: None, matplotlib axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.cm as cm
        
        paths = [polygon.get_path() for polygon in self]
        domain = self.get_domain()[:, :2]

        # Color
        if type(c_poly) == str: # Unicolor
            if c_poly is 'default': c_poly = 'b'
            color_vector = c_poly*len(paths)
            colorbar = False
        else:                  # Colormap
            if cmap is 'default':
                cmap = cm.YlOrRd
            import matplotlib.colors as mcolors
            normalize = mcolors.Normalize(vmin=c_poly.min(), vmax=c_poly.max())
            color_vector = cmap(normalize(c_poly))

        # Plot
        fig = plt.figure(title)
        ax = fig.add_subplot(111)
        for p, c in zip(paths, color_vector):
            ax.add_patch(patches.PathPatch(p, facecolor=c, lw=1, 
                                           edgecolor='k', alpha=alpha))
        ax.set_xlim(domain[0,0],domain[1,0])
        ax.set_ylim(domain[0,1], domain[1,1])
        
        # Colorbar
        if colorbar:
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
            scalarmappaple.set_array(c_poly)
            cbar = plt.colorbar(scalarmappaple, shrink=0.8, aspect=10)
            cbar.ax.set_ylabel(cbar_label, rotation=0)

        if ret: return ax
        
    def iplot(self, c_poly='default', c_holes='c', ret=False, ax=None):
        """
        Improved plot that allows to plot polygons and holes in
        different colors.
        
        :param c_poly: Polygons color.
        :type c_poly: matplotlib color, 'default' or 't' (transparent)
        :param c_holes: Holes color.
        :type c_holes: matplotlib color, 'default' or 't' (transparent)
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :param ax: If a matplotlib axes given, this method will 
            represent the plot on top of this axes. This is used to
            represent multiple plots from multiple geometries, 
            overlapping them recursively.
        :type ax: mplot3d.Axes3D, None
        :returns: None, axes
        :rtype: None, mplot3d.Axes3D
        """
        # Default parameter
        if c_holes == 'default': c_holes = 'c' # cyan for the holes
        
        seed = self.get_seed()
        if c_poly != False:
            ax = Surface(seed['polygons']).plot(color=c_poly, ret=True,
                                                ax=ax)
        if self.holes != [] and c_holes != False:
            ax = Surface(seed['holes']).plot(color=c_holes, ret=True,
                                             ax=ax)
        if ret: return ax
            
    def move(self, d_xyz):
        """
        Translate the Surface in x, y and z coordinates.
        
        :param d_xyz: displacement in x, y(, and z).
        :type d_xyz: tuple (len=2 or 3)
        :returns: ``pyny.Surface``
        """
        return Space(Place(self)).move(d_xyz, inplace=False)[0].surface

    def rotate(self, angle, direction='z', axis=None):
        """
        Returns a new Surface which is the same but rotated about a 
        given axis.
        
        If the axis given is ``None``, the rotation will be computed
        about the Surface's centroid.
        
        :param angle: Rotation angle (in radians)
        :type angle: float
        :param direction: Axis direction ('x', 'y' or 'z')
        :type direction: str
        :param axis: Point in z=0 to perform as rotation axis
        :type axis: tuple (len=2 or 3) or None
        :returns: ``pyny.Surface``
        """
        return Space(Place(self)).rotate(angle, direction, axis)[0].surface

    def mirror(self, axes='x'):
        """
        Generates a symmetry of the Surface respect global axes.
        
        :param axes: 'x', 'y', 'z', 'xy', 'xz', 'yz'...
        :type axes: str
        :returns: ``pyny.Surface``
        """
        return Space(Place(self)).mirror(axes, inplace=False)[0].surface

    def matrix(self, x=(0, 0), y=(0, 0) , z=(0, 0)):
        """
        Copy the ``pyny.Surface`` along a 3D matrix given by the 
        three tuples x, y, z:        

        :param x: Number of copies and distance between them in this
            direction.
        :type x: tuple (len=2)
        :returns: list of ``pyny.Surface``
        """
        space = Space(Place(self)).matrix(x, y, z, inplace=False)
        return [place.surface for place in space]


class Polyhedron(root):
    """
    Represents 3D polygon-based convex polyhedra.
    
    Under the hood, ``pyny.Polyhedron`` class uses the ``pyny.Surface``
    infrastructure to store and operate with the faces (Polygons). This
    ``pyny.Surface`` can be found in ``Polyhedron.aux_surface``.

    Instances of this class work as iterable object. When indexed, 
    returns the ``pyny.Polygons`` which conform it.

    :param polygons: Polygons to be set as the Polyhedron. These 
        Polygons have to be contiguous and form a closed polyhedron\*.
    :type polygons: list of ndarray, list of ``pyny.Polygon``
    :param make_ccw: If True, points will be sorted ccw for each 
        polygon.
    :type make_ccw: bool
    :returns: None
    
    .. note:: \* A concave or open polyhedron will not produce any 
        error and the code will probably work fine but it is important 
        to keep in mind that *pyny3d* was created to work specifically
        with convex and closed bodies and you will probably get errors
        later in other parts of the code.
    
    .. warning:: This object do NOT check the contiguity of the 
        polygons or whether the polyhedron is closed or not, even it is
        actually a requirement.
    """
    def __init__(self, polygons, make_ccw=True, **kwargs):
        self.aux_surface = Surface(polygons, make_ccw=make_ccw)
        self.polygons = self.aux_surface.polygons
    def __iter__(self): return iter(self.polygons)    
    def __getitem__(self, key): return self.polygons[key]

    def seed2pyny(self, seed):
        """
        Re-initialize an object with a seed.
        
        :returns: A new ``pyny.Polyhedron``
        :rtype: ``pyny.Polyhedron``
        """
        return Polyhedron(**seed)

    @staticmethod
    def by_two_polygons(poly1, poly2, make_ccw=True):
        """
        Static method. Creates a closed ``pyny.Polyhedron`` connecting 
        two polygons. Both polygons must have the same number of 
        vertices. The faces of the Polyhedron created have to be planar, 
        otherwise, an error will be raised.
        
        The Polyhedron will have the *poly1* and *poly2* as "top" and 
        "bottom" and the rest of its faces will be generated by matching
        the polygons' vertices in twos.
        
        :param poly1: Origin polygon
        :type poly1: ``pyny.Polygon`` or ndarray (shape=(N, 3))
        :param poly2: Destination polygon
        :type poly2: ``pyny.Polygon`` or ndarray (shape=(N, 3))
        :param make_ccw: If True, points will be sorted ccw for each 
            polygon.
        :type make_ccw: bool
        :returns: Polyhedron
        :rtype: ``pypy.Polyhedron``
        
        .. warning:: If an error is raised, probably the Polyhedron
            have non-planar faces.
        .. warning:: If the Polyhedra are not created with this method
            or ``Place.add_extruded_obstacles()``, holes will not be 
            added.
        """
        if type(poly1) == Polygon:
            poly1 = poly1.points
            poly2 = poly2.points
        
        vertices = np.dstack((poly1, poly2))
        polygons = []
        for i in np.arange(vertices.shape[0])-1:
            polygons.append(np.array([vertices[i, :, 1],
                                      vertices[i+1,:, 1], 
                                      vertices[i+1, :, 0], 
                                      vertices[i,:, 0]]))
        polygons.append(poly1)
        polygons.append(poly2)
        
        return Polyhedron(polygons, make_ccw=make_ccw)

    def get_seed(self):
        """
        Collects the required information to generate a data estructure 
        that can be used to recreate exactly the same geometry object
        via *\*\*kwargs*.
        
        :returns: Object's sufficient info to initialize it.
        :rtype: dict
        """
        return {'polygons': self.aux_surface.get_seed()['polygons']}
        
    def get_plotable3d(self):
        """
        :returns: matplotlib Poly3DCollection
        :rtype: list of mpl_toolkits.mplot3d
        """
        return self.aux_surface.get_plotable3d()
        
    def get_domain(self):
        """
        :returns: opposite vertices of the bounding prism for this 
            object.
        :rtype: ndarray([min], [max])
        
        .. note:: This method automatically stores the solution in order
            to do not repeat calculations if the user needs to call it 
            more than once.
        """
        return self.aux_surface.get_domain()
        
    def get_area(self):
        """
        :returns: The area of the polyhedron.
        """
        return sum([polygon.get_area() for polygon in self.aux_surface])
        
    def move(self, d_xyz):
        """
        Translate the Polyhedron in x, y and z coordinates.
        
        :param d_xyz: displacement in x, y(, and z).
        :type d_xyz: tuple (len=2 or 3)
        :returns: ``pyny.Polyhedron``
        """
        polygon = np.array([[0,0], [0,1], [1,1], [0,1]])
        space = Space(Place(polygon, polyhedra=self))
        return space.move(d_xyz, inplace=False)[0].polyhedra[0]

    def rotate(self, angle, direction='z', axis=None):
        """
        Returns a new Polyhedron which is the same but rotated about a 
        given axis.
        
        If the axis given is ``None``, the rotation will be computed
        about the Polyhedron's centroid.
        
        :param angle: Rotation angle (in radians)
        :type angle: float
        :param direction: Axis direction ('x', 'y' or 'z')
        :type direction: str
        :param axis: Point in z=0 to perform as rotation axis
        :type axis: tuple (len=2 or 3) or None
        :returns: ``pyny.Polyhedron``
        """
        polygon = np.array([[0,0], [0,1], [1,1]])
        space = Space(Place(polygon, polyhedra=self))
        return space.rotate(angle, direction, axis)[0].polyhedra[0]

    def mirror(self, axes='x'):
        """
        Generates a symmetry of the Polyhedron respect global axes.
        
        :param axes: 'x', 'y', 'z', 'xy', 'xz', 'yz'...
        :type axes: str
        :returns: ``pyny.Polyhedron``
        """
        polygon = np.array([[0,0], [0,1], [1,1]])
        space = Space(Place(polygon, polyhedra=self))
        return space.mirror(axes, inplace=False)[0].polyhedra[0]

    def matrix(self, x=(0, 0), y=(0, 0) , z=(0, 0)):
        """
        Copy the ``pyny.Polyhedron`` along a 3D matrix given by the 
        three tuples x, y, z:        

        :param x: Number of copies and distance between them in this
            direction.
        :type x: tuple (len=2)
        :returns: list of ``pyny.Polyhedron``
        """
        polygon = np.array([[0,0], [0,1], [1,1]])
        space = Space(Place(polygon, polyhedra=self))
        space = space.matrix(x, y, z, inplace=False)
        return [place.polyhedra[0] for place in space]


class Place(root):
    """
    Aggregates one ``pyny.Surface``, one Set of points and an indefinite
    number of ``pyny.Polyhedra``.
    
    Represents the union of a surface with an unlimited number of 
    obstacles. All the elements that conform a Place keep their 
    integrity and functionality, what the Place class makes is to give
    the possibility to perform higher level operations in these groups
    of objects.
    
    Instances of this class cannot work as iterable object and cannot be
    indexed.
    
    The lower level instances will be stored in:
        * **Place.surface**
        * **Place.polyhedra**
        * **Place.set_of_points**
    
    :param surface: This is the only necessary input to create a 
        ``pyny.Place``.
    :type surface: ``pyny.Surface``, list of ``pyny.Polygon`` or list
        of ndarray
    :param polyhedra: ``pyny.Polyhedra`` to attach to the 
        ``pyny.Place``.
    :type polyhedra: list of ``pyny.Polyhedra``
    :param set_of_points: Points to attach to the ``pyny.Place``.
    :type set_of_points: ndarray (shape=(N, 3))
    :param make_ccw: If True, points will be sorted ccw for each 
        polygon.
    :type make_ccw: bool
    :returns: None
    
    .. note:: This object is implemented to be used dynamically. Once
        created, it is possible to add elements, with 
        ``.add_set_of_points``, ``.add_extruded_obstacles`` among others,
        without replace it.
    """
    def __init__(self, surface, polyhedra=[], set_of_points=np.empty((0, 3)),
                 make_ccw=True, melt=False, **kwargs):
        # Creating the object
        ## Surface
        if type(surface) == Surface: # Surface object
            self.surface = surface
        elif type(surface) == dict: # Seed
            self.surface = Surface(**surface)
        elif type(surface) == list or type(surface) == np.ndarray: # Simple input
            self.surface = Surface(**{'polygons': surface,
                                      'make_ccw': make_ccw,
                                      'melt': melt})
        else:
            raise ValueError('pyny3d.Place needs a dict or pyny3d.Surface as input')

        ## Polyhedra
        if polyhedra != []:
            if type(polyhedra) != list: polyhedra = [polyhedra]
            if type(polyhedra[0]) == Polyhedron:
                self.polyhedra = polyhedra
            else:
                self.polyhedra = [Polyhedron(polyhedron, make_ccw)
                                  for polyhedron in polyhedra]
        else:
            self.polyhedra = []
        
        ## Set of points
        if type(set_of_points) == np.ndarray:
            if set_of_points.shape[1] == 3:
                self.set_of_points = set_of_points
        else:
            raise ValueError('pyny3d.Place has an invalid set_of_points as input')

    def seed2pyny(self, seed):
        """
        Re-initialize an object with a seed.
        
        :returns: A new ``pyny.Place``
        :rtype: ``pyny.Place``
        """
        # import geoms as pyny
        return Place(**seed)

    def add_set_of_points(self, points):
        """
        Add a new set of points to the existing one without removing it.
        
        :param points: Points to be added.
        :type points: ndarray (shape=(N, 3))
        :returns: None
        """
        self.set_of_points = np.concatenate((self.set_of_points, points))
        
    def get_domain(self):
        """
        :returns: opposite vertices of the bounding prism for this 
            object.
        :rtype: ndarray([min], [max])
        """
        if self.polyhedra != []:
            polyhedras_domain = np.vstack([poly.get_domain() 
                                          for poly in self.polyhedra])
        else:
            polyhedras_domain = np.ones((0, 3))
        points = np.vstack((self.surface.get_domain(), 
                            polyhedras_domain, 
                            self.set_of_points))
        return np.array([points.min(axis=0), points.max(axis=0)])
        
    def get_height(self, points, edge=True, attach=False, 
                   extra_height=0):
        """
        Launch ``pyny.Surface.get_height(points)`` for the Place's 
        Surface.
        
        This method gives the possibility to store the computed points
        along with the Place's set of points. It also makes possible to
        add an extra height (z value) to these points.
        
        The points outside the object will have a NaN value in the
        z column. These point will not be stored but it will be
        returned.
         
        :param points: list of coordinates of the points to calculate.
        :type points: ndarray (shape=(N, 2 or 3))
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :param attach: If True, stores the computed points along with
            the Place's set of points.
        :type attach: bool
        :param extra_height: Adds an extra height (z value) to the
            resulting points.
        :type extra_height: float
        :returns: (x, y, z)
        :rtype: ndarray
        """
        points = self.surface.get_height(points, edge=edge)  
        points[:, 2] += extra_height
        if attach: 
            logic = np.logical_not(np.isnan(points[:, 2]))
            self.add_set_of_points(points[logic])
        else:
            return points

    def mesh(self, mesh_size=1, extra_height=0.1, edge=True, attach=True):
        """
        Generates a set of points distributed in a mesh that covers the 
		whole Place and computes their height.
        
        Generates a xy mesh with a given mesh_size in the 
        Place.surface's domain and computes the Surface's height for the
        nodes. This mesh is alligned with the main directions `x` and 
        `y`.
        
        :param mesh_size: distance between points.
        :type mesh_size: float
        :param extra_height: Adds an extra height (z value) to the
            resulting points.
        :type extra_height: float
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :param attach: If True, stores the computed points along with
            the Place's set of points.
        :type attach: bool
        :returns: (x, y, z)
        :rtype: ndarray
        """
        # Mesh
        a, b = self.get_domain()
        a -= 2*mesh_size # extra bound
        b += 2*mesh_size
        x_mesh = np.arange(a[0], b[0], mesh_size)
        y_mesh = np.arange(a[1], b[1], mesh_size)
        x, y = np.meshgrid(x_mesh, y_mesh)
        xy = np.array([x.ravel(), y.ravel()]).T
        
        # Compute and store
        xyz = self.get_height(xy, edge=edge, attach=attach,
                              extra_height=extra_height)
        if not attach: return xyz
            
    def clear_set_of_points(self):
        """
        Remove all the points in the Place.
        """
        self.set_of_points = np.empty((0, 3))

    def add_holes(self, holes_list, make_ccw=True):
        """
        Add holes to the Place's ``pyny.Surface``.
        
        :param holes_list: Polygons that will be treated as holes.
        :type holes_list: list or ``pyny.Polygon``
        :param make_ccw: If True, points will be sorted ccw.
        :type make_ccw: bool
        :returns: None
        
        .. note:: The holes can be anywhere, not necesarily on the 
            surface.
        """
        self.surface.add_holes(holes_list, make_ccw=make_ccw)

    def add_extruded_obstacles(self, top_polys, make_ccw=True):
        """
        Add polyhedras to the Place by giving their top polygon and
        applying extrusion along the z axis. The resulting polygon
        from the intersection will be declared as a hole in the Surface.
        
        :param top_polys: Polygons to be extruded to the Surface.
        :type top_polys: list of ``pyny.Polygon``
        :param make_ccw: If True, points will be sorted ccw.
        :type make_ccw: bool
        :returns: None

        .. note:: When a top polygon is projected and it
            instersects multiple Surface's polygons, a independent
            polyhedron will be created for each individual 
            intersection\*.
        .. warning:: The top polygons have to be over the Surface, that
            is, their z=0 projection have to be inside of Surface's z=0 
            projection.
        .. warning:: If the Polyhedra are not created with this method
            or ``Polyhedron.by_two_polygons()``, holes will not be 
            added.
        """
        if type(top_polys) != list: top_polys = [top_polys]
        for poly1 in top_polys:
            if type(poly1) != Polygon:
                obstacle = Polygon(poly1, make_ccw)
            intersections_dict = self.surface.intersect_with(obstacle)
            
            base = []
            for i, xy in intersections_dict.items():
                base.append(self.surface[i].get_height(xy, full=True))
            base_surf = Surface(base)
            base_surf.melt()
            
            for base_poly in base_surf:
                obst_points = obstacle.get_height(base_poly.points, 
                                                full=True)
                self.surface.holes.append(base_poly)
                self.polyhedra.append(Polyhedron.by_two_polygons(
                                      base_poly.points, 
                                      obst_points, 
                                      make_ccw))
            
    def get_seed(self):
        """
        Collects the required information to generate a data estructure 
        that can be used to recreate exactly the same geometry object
        via *\*\*kwargs*.
        
        :returns: Object's sufficient info to initialize it.
        :rtype: dict
        """
        seed = {}
        seed['surface'] = self.surface.get_seed()
        polyhedra = [polyhedron.get_seed()['polygons'] 
                     for polyhedron in self.polyhedra]
        if not polyhedra: polyhedra = []
        seed['polyhedra'] = polyhedra
        
        if self.set_of_points.shape[0] != 0:
            seed['set_of_points'] = self.set_of_points 
        else:
            seed['set_of_points'] = np.empty((0, 3))
        return seed

    def get_plotable3d(self):
        """
        :returns: matplotlib Poly3DCollection
        :rtype: list of mpl_toolkits.mplot3d
        """
        polyhedra = sum([polyhedron.get_plotable3d() 
                         for polyhedron in self.polyhedra], [])
        return polyhedra + self.surface.get_plotable3d()

    def iplot(self, c_poly='default', c_holes='default', c_sop='r',
              s_sop=25, extra_height=0, ret=False, ax=None):
        """
        Improved plot that allows to plot polygons and holes in
        different colors and to change the size and the color of the
        set of points.
        
        The points can be plotted accordingly to a ndarray colormap.
        
        :param c_poly: Polygons color.
        :type c_poly: matplotlib color, 'default' or 't' (transparent)
        :param c_holes: Holes color.
        :type c_holes: matplotlib color, 'default' or 't' (transparent)
        :param c_sop: Set of points color.
        :type c_sop: matplotlib color or colormap
        :param s_sop: Set of points size.
        :type s_sop: float or ndarray
        :param extra_height: Elevates the points in the visualization.
        :type extra_height: float
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :param ax: If a matplotlib axes given, this method will 
            represent the plot on top of this axes. This is used to
            represent multiple plots from multiple geometries, 
            overlapping them recursively.
        :type ax: mplot3d.Axes3D, None
        :returns: None, axes
        :rtype: None, mplot3d.Axes3D
        """
        ax = self.surface.iplot(c_poly=c_poly, c_holes=c_holes,
                                ret=True, ax=ax)
        for polyhedron in self.polyhedra:
            ax = polyhedron.plot(color=c_poly, ret=True, ax=ax)
        if c_sop != False:
            p = self.set_of_points
            ax.scatter(p[:, 0], p[:, 1], p[:, 2]+extra_height, 
                       c=c_sop, s=s_sop)
        self.center_plot(ax)
        if ret: return ax

    def move(self, d_xyz):
        """
        Translate the Place in x, y and z coordinates.
        
        :param d_xyz: displacement in x, y(, and z).
        :type d_xyz: tuple (len=2 or 3)
        :returns: ``pyny.Place``
        """
        return Space(self).move(d_xyz, inplace=False)[0]

    def rotate(self, angle, direction='z', axis=None):
        """
        Returns a new Place which is the same but rotated about a 
        given axis.
        
        If the axis given is ``None``, the rotation will be computed
        about the Place's centroid.
        
        :param angle: Rotation angle (in radians)
        :type angle: float
        :param direction: Axis direction ('x', 'y' or 'z')
        :type direction: str
        :param axis: Point in z=0 to perform as rotation axis
        :type axis: tuple (len=2 or 3) or None
        :returns: ``pyny.Place``
        """
        return Space(self).rotate(angle, direction, axis)[0]

    def mirror(self, axes='x'):
        """
        Generates a symmetry of the Place respect global axes.
        
        :param axes: 'x', 'y', 'z', 'xy', 'xz', 'yz'...
        :type axes: str
        :returns: ``pyny.Place``
        """
        return Space(self).mirror(axes, inplace=False)[0]

    def matrix(self, x=(0, 0), y=(0, 0) , z=(0, 0)):
        """
        Copy the ``pyny.Place`` along a 3D matrix given by the 
        three tuples x, y, z:        

        :param x: Number of copies and distance between them in this
            direction.
        :type x: tuple (len=2)
        :returns: list of ``pyny.Place``
        """
        space = Space(self).matrix(x, y, z, inplace=False)
        return [place for place in space]


class Space(root):
    """
    the highest level geometry class. It Aggregates ``pyny.Places`` to 
    group computations. It can be initialized empty.
    
    The lower level instances will be stored in:
        * **Space.places**

    :param places: Places or empty list.
    :type places: list of ``pyny.Place``
    :returns: None
    
    Instances of this class work as iterable object. When indexed, 
    returns the ``pyny.Places`` which conform it.
    
    .. note:: This class is implemented to be used dynamically. Once
        created, it is possible to add elements, with 
        ``.add_places``, ``.add_spaces`` among others, without replace 
        it.
    .. warning:: Although it is a dynamic class, it is recommended to 
        use the methods to manipulate it. Editing the internal 
        attributes or methods directly can result in a bad behavior.
    """
    def __init__(self, places=[], **kwargs):
        # Empty initializations
        self.places = []
        
        # Lock attributes
        self.locked = False
        self.map = None
        self.seed = None
        self.map2seed_schedule = None
        self.explode_map_schedule = None
        
        # Creating the object
        if places != []:
            if type(places) != list: places = [places]
            if type(places[0]) == Place: # Places already initialized
                self.add_places(places)
            elif type(places[0]) == dict: # Initialize the places
                self.add_places([ Place(**place) for place in places ])
            else:
                raise ValueError('pyny3d.Space needs a list, dict or '+\
                                 'pyny3d.Place as input')

    def __iter__(self): return iter(self.places)
    def __getitem__(self, key): return self.places[key]
        
    def lock(self):
        """
        Precomputes some parameters to run faster specific methods like
        Surface.classify. This method is automatically launched before shadows
        computation.
		
        :returns: None
        """
        if self.locked: return
        from pyny3d.utils import bool2index
        # seed
        self.seed = self.get_seed()
        
        # map
        self.map = self.get_map()

        # map2polygons schedule
        m2p = [[], [], 0] # [polygons, holes, sop]
        index, points = self.get_map()
                
        ## points
        bool_1 = index[:, 1] == -1
        m2p[2] = bool2index(bool_1)
        index_1 = bool2index(np.logical_not(bool_1)) # remain
        index = index[index_1]
        
        ## new index
        index_bool = np.diff(index[:, 2]*1e12
                            +index[:, 1]*1e8 
                            +index[:, 0]*1e4) != 0
        
        ## Dissemination loop
        dif = np.arange(index_bool.shape[0], dtype=int)[index_bool]+1
        dif = np.append(dif, index_bool.shape[0]+1)
        i = 0
        for j in dif:
            if index[i, 2] < 0: # hole
                m2p[1].append(index_1[np.arange(i, j)])
            if index[i, 2] >= 0: # polygon
                m2p[0].append(index_1[np.arange(i, j)])
            i = j
        self.explode_map_schedule = m2p

        # Sort areas
        areas = []
        for poly in self.explode()[0]:
            areas.append(Polygon(poly, False).get_area())
        self.sorted_areas = np.argsort(np.array(areas))[::-1]
        
        # Lock
        self.locked = True

    def seed2pyny(self, seed):
        """
        Re-initialize an object with a seed.
        
        :returns: A new ``pyny.Space``
        :rtype: ``pyny.Space``
        
        .. seealso:: 
            
            * :func:`get_seed` 
            * :func:`get_map`
            * :func:`map2seed` 
            * :func:`explode_map`
            
        """
        return Space(**seed)
        
    def add_places(self, places, ret=False):
        """
        Add ``pyny.Places`` to the current space.
        
        :param places: Places to add.
        :type places: list of pyny.Place
        :param ret: If True, returns the whole updated Space. 
        :type ret: bool
        :returns: None, ``pyny.Space``
        
        .. warning:: This method acts inplace.
        """
        if type(places) != list: places = [places]        
        self.places += places
        if ret: return self

    def add_spaces(self, spaces, ret=False):
        """
        Add ``pyny.Spaces`` to the current space. In other words, it 
		merges multiple ``pyny.Spaces`` in this instance.
        
        :param places: ``pyny.Spaces`` to add.
        :type places: list of pyny.Spaces
        :param ret: If True, returns the whole updated Space. 
        :type ret: bool
        :returns: None, ``pyny.Space``
        
        .. warning:: This method acts inplace.
        """
        if type(spaces) != list: spaces = [spaces]
        Space.add_places(self, sum([space.places for space in spaces], []))
        if ret: return self

    def get_domain(self):
        """
        :returns: opposite vertices of the bounding prism for this 
            object.
        :rtype: ndarray([min], [max])
        """
        points = np.vstack([place.get_domain() for place in self])
        return np.array([points.min(axis=0), points.max(axis=0)])

    def get_seed(self):
        """
        Collects the required information to generate a data estructure 
        that can be used to recreate exactly the same geometry object
        via *\*\*kwargs*.
        
        :returns: Object's sufficient info to initialize it.
        :rtype: dict
        
        .. seealso:: 
            
            * :func:`get_map` 
            * :func:`map2pyny`
            * :func:`map2seed` 
            * :func:`explode_map`
            

        """
        self.seed = {'places': [place.get_seed() for place in self]}
        return self.seed

    def get_plotable3d(self):
        """
        :returns: matplotlib Poly3DCollection
        :rtype: list of mpl_toolkits.mplot3d
        """
        return sum([place.get_plotable3d() for place in self], [])
        
    def get_sets_of_points(self):
        """
        Collects all the sets of points for the Places contained in the 
		Space.
        
        :returns: An array with the points of all ``pyny.Places`` which 
            form this ``pyny.Space``.
        :rtype: ndarray (shape=(N, 3))
        """
        return np.concatenate([place.set_of_points for place in self])
        
    def get_sets_index(self):
        """
        Returns a one dimension array with the Place where the points 
        belong.
        
        :returns: The ``pyny.Place`` where the points belong.
        :rtype: list of int
        """
        index = []
        for i, place in enumerate(self):
            index.append(np.ones(place.set_of_points.shape[0])*i)
        return np.concatenate(index).astype(int)

    def get_polygons(self):
        """
        Collects all polygons for the Places in the Space.
        
        :returns: The polygons which form the whole ``pyny.Space``.
        :rtype: list of ``pyny.Polygon``
        """
        return np.concatenate([place.set_of_points for place in self])
        
    def clear_sets_of_points(self):
        """
        Clears all the sets of points of each ``pyny.Place`` in the 
        Space.
        """
        for place in self: place.set_of_points = np.ones((0, 3))

    def get_map(self):
        """
        Collects all the points coordinates from this ``pyny.Space``
        instance.
        
        In order to keep the reference, it returns an index with the
        following key:
        
            * The first column is the Place.
            * The second column is the body (-1: points, 0: surface, 
                                             n: polyhedron)
            * The third column is the polygon (-n: holes)
            * The fourth column is the point.
 
        :returns: [index, points]
        :rtype: list of ndarray

        .. note:: This method automatically stores the solution in order
            to do not repeat calculations if the user needs to call it 
            more than once.

        .. seealso:: 
            
            * :func:`get_seed` 
            * :func:`map2pyny`
            * :func:`map2seed` 
            * :func:`explode_map`
            
        """
        seed = self.get_seed()['places'] # template
        
        points = []
        index = []
        for i, place in enumerate(seed):
            # Set of points [_, -1, 0, _]
            n_points = place['set_of_points'].shape[0]
            if n_points != 0: # It can be False (no set_of_points)
                points.append(place['set_of_points'])
                index.append(np.vstack((np.tile(np.array([[i], [-1], [0]]), 
                                                         n_points), 
                                        np.arange(n_points))))
            #Holes [_, 0, -N, _]
            for ii, hole in enumerate(place['surface']['holes']):
                n_points = hole.shape[0]
                points.append(hole)
                index.append(np.vstack((np.tile(np.array([[i], [0], [-ii-1]]), 
                                                         n_points), 
                                        np.arange(n_points))))
            #Surface [_, 0, N, _]
            for ii, polygon in enumerate(place['surface']['polygons']):
                n_points = polygon.shape[0]
                points.append(polygon)
                index.append(np.vstack((np.tile(np.array([[i], [0], [ii]]), 
                                                         n_points), 
                                        np.arange(n_points))))
            #Polyhedras [_, N, _, _]
            if len(place['polyhedra']) != 0: # It can be False (no obstacles)
                for iii, polygon_list in enumerate(place['polyhedra']):
                    for iv, polygon in enumerate(polygon_list):
                        n_points = polygon.shape[0]
                        points.append(polygon)
                        index.append(np.vstack((np.tile(np.array([[i], [1+iii], 
                                                                  [iv]]), n_points), 
                                                np.arange(n_points))))

        index = np.concatenate(index, axis=1).T
        points = np.concatenate(points)
        self.map = [index, points]
        return self.map
 
    def map2seed(self, map_):
        """
        Returns a seed from an altered map. The map needs to have the 
        structure of this ``pyny.Space``, that is, the same as
        ``self.get_map()``.
        
        :param map_: the points, and the same order, that appear at 
            ``pyny.Space.get_map()``.
        :type map_: ndarray (shape=(N, 3))
        :returns: ``pyny.Space`` seed.
        :rtype: dict
        
        .. seealso:: 
            
            * :func:`get_seed` 
            * :func:`get_map` 
            * :func:`map2pyny`
            * :func:`explode_map`
            
        """
        seed = self.get_seed()['places'] # Template
        
        o = 0
        for i, place in enumerate(seed):
            # Set of points [_, -1, 0, _]
            if place['set_of_points'].shape[0] != 0: # Maybe no set_of_points
                polygon = place['set_of_points']
                seed[i]['set_of_points'] = map_[o: o + polygon.shape[0], :]
                o += polygon.shape[0]
            #Holes [_, 0, -N, _]
            for ii, hole in enumerate(place['surface']['holes']):
                seed[i]['surface']['holes'][ii] = map_[o: o + hole.shape[0], :]
                o += hole.shape[0]
            #Surface [_, 0, N, _]
            for ii, polygon in enumerate(place['surface']['polygons']):
                seed[i]['surface']['polygons'][ii] = map_[o: o + 
                                                          polygon.shape[0], :]
                o += polygon.shape[0]
            #Polyhedras [_, N, _, _]
            if len(place['polyhedra']) != 0: # Maybe no polyhedra
                for ii, polygon_list in enumerate(place['polyhedra']):
                    for iii, polygon in enumerate(polygon_list):
                        seed[i]['polyhedra'][ii][iii] = map_[o: o +
                                                        polygon.shape[0], :]
                        o += polygon.shape[0]
        return {'places': seed}

    def map2pyny(self, map_):
        """
        Returns a different version of this ``pyny.Space`` using an 
		altered map.
        
        :param map_: the points, and the same order, that appear at 
            ``pyny.Space.get_map()``.
        :type map_: ndarray (shape=(N, 3))
        :returns: ``pyny.Space``
        
        .. seealso:: 
            
            * :func:`get_seed` 
            * :func:`get_map` 
            * :func:`map2seed`
            * :func:`explode_map`
            
        """
        return self.seed2pyny(self.map2seed(map_))
        
    def explode(self):
        """
        Collects all the polygons, holes and points in the Space 
        packaged in a list. The returned geometries are not in *pyny3d*
        form, instead the will be represented as *ndarrays*.
        
        :returns: The polygons, the holes and the points.
        :rtype: list
        """        
        seed = self.get_seed()['places']
        
        points = []
        polygons = []
        holes = []
        for place in seed:
            points.append(place['set_of_points'])
            polygons += sum(place['polyhedra'], [])
            polygons += place['surface']['polygons']
            holes += place['surface']['holes']
        return [polygons, holes, np.concatenate(points, axis=0)]
        
    def explode_map(self, map_):
        """
        Much faster version of ``pyny.Space.explode()`` method for 
        previously locked ``pyny.Space``.
        
        :param map_: the points, and the same order, that appear at 
            ``pyny.Space.get_map()``. There is no need for the index if 
            locked.
        :type map_: ndarray (shape=(N, 3))
        :returns: The polygons, the holes and the points.
        :rtype: list
        
        .. seealso:: 
            
            * :func:`get_seed` 
            * :func:`get_map` 
            * :func:`map2pyny`
            * :func:`map2seed`
            
        """        
        if self.explode_map_schedule is None:
            index = map_[0]
            points = map_[1]
            
            # points
            k = index[:, 1] == -1
            sop = points[k] # Set of points
            index = index[np.logical_not(k)]
            points = points[np.logical_not(k)]
            
            # new index
            index_bool = np.diff(index[:, 2]*1e12
                                +index[:, 1]*1e8 
                                +index[:, 2]*1e4).astype(bool)

            # Dissemination loop
            polygons = []
            holes = []
            dif = np.arange(index_bool.shape[0], dtype=int)[index_bool]+1
            dif = np.append(dif, index_bool.shape[0]+1)
            i = 0
            for j in dif:
                if index[i, 2] < 0: # hole
                    holes.append(points[i:j, :])
                if index[i, 2] >= 0: # polygon
                    polygons.append(points[i:j, :])
                i = j
            return [polygons, holes, sop]
        else:
            # Only points (without index) allowed
            if type(map_) == list:
                points = map_[1]
            else:
                points = map_
            ex = self.explode_map_schedule
            polygons = [ points[p ,:] for p in ex[0] ]
            holes = [ points[p ,:] for p in ex[1] ]
            sop = points[ex[2] ,:]
            return [polygons, holes, sop]
            
    def get_height(self, points, edge=True, attach=False, extra_height=0):
        """
        Launch ``pyny.Place.get_height(points)`` recursively for all 
        the ``pyny.Place``.
        
        The points outside the object will have a NaN value in the
        z column. These point will not be stored but it will be
        returned.
         
        :param points: list of coordinates of the points to calculate.
        :type points: ndarray (shape=(N, 2 or 3))
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :param attach: If True, stores the computed points along with
            the Place's set of points.
        :type attach: bool
        :param extra_height: Adds an extra height (z value) to the
            resulting points.
        :type extra_height: float
        :returns: (x, y, z)
        :rtype: ndarray
        """
        for place in self:
            points = place.get_height(points, edge, attach, extra_height)
        if not attach: return points
        
    def mesh(self, mesh_size=1, extra_height=0.1, edge=True, attach=True):
        """
        Launch `pyny.Place.mesh(points)` recursively for all 
        the ``pyny.Place`` individually.
        
        :param mesh_size: distance between points.
        :type mesh_size: float
        :param extra_height: Adds an extra height (z value) to the
            resulting points.
        :type extra_height: float
        :param edge: If True, consider the points in the Polygon's edge
            inside the Polygon.
        :type edge: bool
        :param attach: If True, stores the computed points along with
            the Place's set of points.
        :type attach: bool
        :returns: (x, y, z)
        :rtype: ndarray
        """
        for place in self:
            place.mesh(mesh_size, extra_height, edge, attach)
        
    def move(self, d_xyz, inplace=False):
        """
        Translate the whole Space in x, y and z coordinates.
        
        :param d_xyz: displacement in x, y(, and z).
        :type d_xyz: tuple (len=2 or 3)
        :param inplace: If True, the moved ``pyny.Space`` is copied and 
            added to the current ``pyny.Space``. If False, it returns 
            the new ``pyny.Space``.
        :type inplace: bool
        :returns: None, ``pyny.Space``
        """
        state = Polygon.verify
        Polygon.verify = False
        if len(d_xyz) == 2: d_xyz = (d_xyz[0], d_xyz[1], 0)
        xyz = np.array(d_xyz)
        
        # Add (dx, dy, dz) to all the coordinates
        map_ = self.get_map()[1] + xyz
        space = self.map2pyny(map_)
        
        Polygon.verify = state
        if inplace:
            self.add_spaces(space)
            return None
        else:
            return space

    def rotate(self, angle, direction='z', axis=None):
        """
        Returns a new Space which is the same but rotated about a 
        given axis.
        
        If the axis given is ``None``, the rotation will be computed
        about the Space's centroid.
        
        :param angle: Rotation angle (in radians)
        :type angle: float
        :param direction: Axis direction ('x', 'y' or 'z')
        :type direction: str
        :param axis: Point in z=0 to perform as rotation axis
        :type axis: tuple (len=2 or 3) or None
        :returns: ``pyny.Space``
        """
        state = Polygon.verify
        Polygon.verify = False
        if axis is None:
            axis = self.get_centroid()
        else:
            if len(axis) == 2: axis = np.array([axis[0], axis[1], 0])
        map_ = self.get_map()[1] - axis
        
        ## Rotation matrix
        c = np.cos(angle)
        s = np.sin(angle)
        if direction == 'z':
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        elif direction == 'y':
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif direction == 'x':
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            
        rotated_ = np.dot(R, map_.T).T
        map_ = rotated_ + axis
        space = self.map2pyny(map_)
        Polygon.verify = state
        return space

    def matrix(self, x=(0, 0), y=(0, 0) , z=(0, 0), inplace=True):
        """
        Copy the ``pyny.Space`` along a 3D matrix given by the three
        tuples x, y, z:        

        :param x: Number of copies and distance between them in this
            direction.
        :type x: tuple (len=2)
        :param inplace: If True, the moved ``pyny.Space`` is copied and 
            added to the current ``pyny.Space``. If False, it returns 
            the new ``pyny.Space``.
        :type inplace: bool        
        :returns: None, ``pyny.Space``
        """
        state = Polygon.verify
        Polygon.verify = False
        base = False
        if x[0] != 0:
            x = (np.ones(x[0]-1)*x[1]).cumsum()
            if not base: base = self.copy()
            single = self.copy()
            for pos in x:
                base.add_spaces(single.move((pos, 0, 0), inplace=False))
        if y[0] != 0:
            y = (np.ones(y[0]-1)*y[1]).cumsum()
            if not base: base = self.copy()
            single = base.copy()
            for pos in y:
                base.add_spaces(single.move((0, pos, 0), inplace=False))
        if z[0] != 0:           
            z = (np.ones(z[0]-1)*z[1]).cumsum()
            if not base: base = self.copy()
            single = base.copy()
            for pos in z:
                base.add_spaces(single.move((0, 0, pos), inplace=False))
        
        Polygon.verify = state
        if inplace:
            self.places = []
            self.add_spaces(base)
        else:
            return base
            
    def mirror(self, axes='x', inplace=False):
        """
        Generates a symmetry of the Space respect global axes.
        
        :param axes: 'x', 'y', 'z', 'xy', 'xz', 'yz'...
        :type axes: str
        :param inplace: If True, the new ``pyny.Space`` is copied and 
            added to the current ``pyny.Space``. If False, it returns 
            the new ``pyny.Space``.
        :type inplace: bool        
        :returns: None, ``pyny.Space``
        """
        state = Polygon.verify
        Polygon.verify = False
        mirror = np.ones(3)
        if 'x' in axes:
            mirror *= np.array([-1, 1, 1])
        if 'y' in axes:
            mirror *= np.array([1, -1, 1])
        if 'z' in axes:
            mirror *= np.array([1, 1, -1])
        
        map_ = self.get_map()[1] * mirror
        space = self.map2pyny(map_)
        
        Polygon.verify = state
        if inplace:
            self.add_spaces(space)
            return None
        else:
            return space

    def photo(self, azimuth_zenit, plot=False):
        """
        Computes a change of the reference system for the whole 
        ``pyny.Space`` to align the `y` axis with a given direction. 
        Returns its elements (polygons, holes, points) extracted in a
        list.
        
        In its conception, this method was created as a tool for the 
        shadows computation to calculate "what is in front and what 
        is behind to the look of the Sun". For this reason, the 
        direction is given in spherical coordinates by two angles: the 
        azimth and the zenit.
        
            * The azimuth is zero when pointing to the South, -pi/4 to 
              the East, pi/4 to the West and pi/2 to the North.
            * The zenit is zero at the ground level and pi/4 "pointing
              completely orthogonal to the sky".
        
        In short, this methods answer "How would the ``pyny.Space`` look
        in a photograph taken from an arbitrary direction in cylindrical 
        perpective?"
        
        The photograph has a new reference system: x, y, depth. The sign
        of the new depth coordinate has to be checked before assuming 
        what is closer and what is further inasmuch as it changes
        depending on the direction of the photo.
        
        :param azimuth_zenit: Direction of the photo in spherical 
            coordinates and in radians.
        :type azimuth_zenit: tuple
        :param plot: If True, is shows the photo visualization.
        :type plot: bool
        :returns: Exploded ``pyny.Space``
        :rtype: list
        
        .. note:: Before assume that this method do exactly what it is 
            supposed to do, it is highly recommended to visualy verify 
            throught the *plot=True* argument. It is easy to introduce
            the angles in a different sign criteria, among other 
            frequent mistakes.
        """
        self.lock()
        
        a, z = azimuth_zenit
        R = np.array([[np.cos(a), -np.sin(a)*np.cos(z), np.sin(z)*np.sin(a)],
                      [np.sin(a), np.cos(z)*np.cos(a), -np.cos(a)*np.sin(z)],
                      [0, np.sin(z), np.cos(z)]])
        _, points = self.map
        G = np.dot(R, points.T).T # Here it is in self.Space coordinates
            
        # Coordinate change
        G = np.array([G[:,0], G[:,2], G[:,1]]).T # Photograph coordinate
        poly_hole_points = self.explode_map(G)
        
        if plot:
            polygons, holes, points = poly_hole_points
            aux_surface = Surface(polygons, holes=holes, make_ccw=False)
            ax = aux_surface.plot2d(alpha=0.6, ret=True)
            if points.shape[0] > 0: 
                ax.scatter(points[:, 0], points[:, 1], c='#990000', s=25)
        return poly_hole_points
            
    def iplot(self, places=-1, c_poly='default', c_holes='default', 
              c_sop='r', s_sop=25, extra_height=0, ret=False, ax=None):
        """
        Improved plot that allows to visualize the Places in the Space
        selectively. It also allows to plot polygons and holes in
        different colors and to change the size and the color of the
        set of points.
        
        The points can be plotted accordingly to a ndarray colormap.
        
        :param places: Indexes of the Places to visualize.
        :type places: int, list or ndarray
        :param c_poly: Polygons color.
        :type c_poly: matplotlib color, 'default' or 't' (transparent)
        :param c_holes: Holes color.
        :type c_holes: matplotlib color, 'default' or 't' (transparent)
        :param c_sop: Set of points color.
        :type c_sop: matplotlib color or colormap
        :param s_sop: Set of points size.
        :type s_sop: float or ndarray
        :param ret: If True, returns the figure. It can be used to add 
            more elements to the plot or to modify it.
        :type ret: bool
        :param ax: If a matplotlib axes given, this method will 
            represent the plot on top of this axes. This is used to
            represent multiple plots from multiple geometries, 
            overlapping them recursively.
        :type ax: mplot3d.Axes3D, None
        :returns: None, axes
        :rtype: None, mplot3d.Axes3D
        """
        if places == -1: 
            places = range(len(self.places))
        elif type(places) == int:
            places = [places]
            
        places = np.array(places)
        places[places<0] = len(self.places) + places[places<0]
        places = np.unique(places)
        
        aux_space = Space([self[i] for i in places])
        for place in aux_space:
            ax = place.iplot(c_poly, c_holes, c_sop, s_sop, extra_height,
                             ret=True, ax=ax)
        aux_space.center_plot(ax)
        if ret: return ax

    def shadows(self, data=None, t=None, dt=None, latitude=None,
                init='empty', resolution='mid'):
        '''
        Initializes a ShadowManager object for this ``pyny.Space`` 
        instance.
        
        The 'empty' initialization accepts ``data`` and ``t`` and ``dt``
        but the ShadowsManager will not start the calculations. It will
        wait the user to manually insert the rest of the parameters.
        Call ``ShadowsManager.run()`` to start the shadowing 
        computations.
        
        The 'auto' initialization pre-sets all the required parameters 
        to run the computations\*. The available resolutions are:
        
            * 'low'
            * 'mid'
            * 'high'
            
        The 'auto' mode will use all the arguments different than 
        ``None`` and the ``set_of_points`` of this ``pyny.Space`` if 
        any.
            
        :param data: Data timeseries to project on the 3D model 
            (radiation, for example).
        :type data: ndarray (shape=N), None
        :param t: Time vector in absolute minutes or datetime objects
        :type t: ndarray or list, None
        :param dt: Interval time to generate t vector.
        :type dt: int, None
        :param latitude: Local latitude.
        :type latitude: float (radians)
        :param init: Initialization mode
        :type init: str
        :param init: Resolution for the time vector generation (if 
            ``None``), for setting the sensible points and for the 
            Voronoi diagram.
        :type init: str
        :returns: ``ShadowsManager`` object
        '''
        from pyny3d.shadows import ShadowsManager
        
        if init == 'auto':
            # Resolution
            if resolution == 'low':
                factor = 20
            elif resolution == 'mid':
                factor = 40
            elif resolution == 'high':
                factor = 70
            if dt is None: dt = 6e4/factor
            if latitude is None: latitude = 0.65
            
            # Autofill ShadowsManager Object
            sm = ShadowsManager(self, data=data, t=t, dt=dt, 
                                latitude=latitude)
            if self.get_sets_of_points().shape[0] == 0:
                max_bound = np.diff(self.get_domain(), axis=0).max()
                sm.space.mesh(mesh_size=max_bound/factor, edge=True)
            ## General parameters
            sm.arg_vor_size = 3.5/factor
            sm.run()
            return sm
            
        elif init == 'empty':
            return ShadowsManager(self, data=data, t=t, dt=dt, 
                                  latitude=latitude)
