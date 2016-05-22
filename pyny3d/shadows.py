# -*- coding: utf-8 -*-
import numpy as np
import pyny3d.geoms as pyny

class ShadowsManager(object):
    """
    Class in charge of the management for the shadows simulations.
    
    It can be initialize as standalone object or associated to a 
    ``pyny.Space`` through the ``.shadow`` method.
    
    The only argument needed for the simulator to run is ``t`` or ``dt``
    and the ``latitude``. If the ShadowsManager is initialized from
    ``pyny.Space.shadows`` it is possible to run the execution in *auto*
    mode without inputing anything.
    
    Some explanaions about how it works:
    
    The shadows are computed discretely using a set of distributed
    **sensible points** through the model. These points can be set with 
    the ``.get_height(attach=True)`` or the ``.mesh()`` methods.
    
    At the same time, the sun positions are also discretized. The 
    simulator needs a finite number of positions, given by their azimuth
    and zenit. Anyway, it is more convenient to give it a time vector 
    and the latitude and let the program calculate the sun positions for 
    you.
    
    For convenience, the time is managed in "absolute minutes" within 
    the range of a year in the computations, that is, the first possible
    interval [0] is the Jan 1 00:00 and the last [525599] is Dec 31 
    23:59. February 29 is not taken into account. It is possible to 
    automatically create an equally spaced t vector by giving a fixed
    interval, althought the inputed vectors an be irregular.
    
    In view of the fact that there are, potentially, more than 8000 
    sunnys half-hour intervals in an year, the program precomputes a 
    discretization for the Solar Horizont (azimuth, zenit pairs) and 
    classify the *t* and *data* vectors. The goal is to approximate 
    these 8000 interval simulations to a less than 340 with an maximum 
    error of 3 deg (0.05rads).
    
    This discretization is manually\* adjustable to be able to fastly
    compute large datasets at low resolution before the serious 
    computations start.
    
    For now, the Solar Horizont discretization can only be automatically 
    computed by a mesh. In the future more complex and convenient
    discretizations will be available. Anyway, it is possible to input
    a custom discretization by manually introducing the atributtes 
    described in :func:`Voronoi_SH`.
        
    Finally,    
    
    the atributes which can be safely manipulated to tune up the 
    simulator before the computations are all which start with *arg_* 
    (= default values):

        * .arg_data
        * .arg_t
        * .arg_dt
        * .arg_latitude = None
        * .arg_run_true_time = False
        * .arg_longitude = None (only for ``true_time``)
        * .arg_UTC = None (only for ``true_time``)
        * .arg_zenitmin = 0.1 (minimum zenit, avoid irrelevant errors 
            from trigonometric approximations)
        * .arg_vor_size = 0.15 (mesh_size of the Voronoi diagram)
    
    :param space: 3D model to run the simulation.
    :type space: ``pyny.Space`` 
    :param data: Data timeseries to project on the 3D model (radiation,
        for example).
    :type data: ndarray (shape=N), None
    :param t: Time vector in absolute minutes or datetime objects
    :type t: ndarray or list, None
    :param dt: Interval time to generate t vector.
    :type dt: int, None
    :param latitude: Local latitude.
    :type latitude: float (radians)
    :returns: None
        
    .. note:: \* In the future, the discretizations will be 
        automated based on error adjustment.
    
    .. warning:: The shadows computation do not take care
        of the holes\*, instead, they can be emulated by a collection of 
        polygons.
    """
    def __init__(self, space, data=None, t=None, dt=None, latitude=None):
        from pyny3d.shadows import Viz
        self.viz = Viz(self)
        
        self.space = space
        # Arguments
        self.arg_data = data
        self.arg_t = t
        self.arg_dt = dt
        self.arg_latitude = latitude
        self.arg_run_true_time = False
        self.arg_longitude = None
        self.arg_UTC = None
        self.arg_zenitmin = 0.05
        self.arg_vor_size = 0.15
        
        # Processed information
        ## Precalculations
        self.diff_t = None
        self.integral = None
        
        ## Voronoi
        self.t2vor_map = None
        self.vor_freq = None
        self.vor_surf = None
        self.vor_centers = None
        
        ## get_sunpos
        self.azimuth_zenit = None
        self.true_time = None
        
        ## compute_shadows
        self.light_vor = None
        
        ## project_data
        self.proj_vor = None
        self.proj_points = None

    def run(self):
        """
        Run the shadowing computation with the values stored in 
        ``self.arg_``. Precomputed information is stored in:
        
            * **.diff_t** (*ndarray*): ``np.diff(t)``
            * **.integral** (*ndarray*): Trapezoidal data integration 
              over time.

        The steps are:
        
            * :func:`get_sunpos`
            * :func:`Vonoroi_SH`
            * :func:`compute_shadows`
            * :func:`project_data`
        
        :retruns: None
        """
        # Adapt series
        ## time
        if self.integral is None:
            if self.arg_t is not None:
                import datetime
                if type(self.arg_t[0]) == datetime.datetime:
                    self.arg_t = self.to_minutes(time_obj=self.arg_t)
                else:
                    self.arg_t = np.round(self.arg_t)
            elif self.arg_dt is not None:
                self.arg_dt = np.round(self.arg_dt)
                self.arg_t = self.to_minutes(dt=self.arg_dt)
            else:
                raise ValueError('At least one time parameter is needed.')
            self.diff_t = np.diff(self.arg_t)
        
        ## data
        if self.arg_data is None:
            self.arg_data = np.ones(self.arg_t.shape[0])
        dt = self.diff_t/60  # hs
        rect = self.arg_data[:-1]/1000*dt  # kilounits
        triang_side = np.diff(self.arg_data)
        triang = 0.5*triang_side*dt
        self.integral = rect + triang
        self.integral = np.hstack((0, self.integral))

        # Computation
        if self.azimuth_zenit is None:
            self.get_sunpos(self.arg_t, self.arg_run_true_time)
        if self.vor_centers is None:
            self.Vonoroi_SH(self.arg_vor_size)
        self.compute_shadows()
        self.project_data()
    
    def Vonoroi_SH(self, mesh_size=0.1):
        """
        Generates a equally spaced mesh on the Solar Horizont (SH).
        
        Computes the Voronoi diagram from a set of points given by pairs
        of (azimuth, zenit) values. This discretization completely
        covers all the Sun positions.
        
        The smaller mesh size, the better resolution obtained. It is 
        important to note that this heavily affects the performance.
        
        The generated information is stored in:
            * **.t2vor_map** (*ndarray*): Mapping between time vector and
              the Voronoi diagram.
            * **.vor_freq** (*ndarray*): Number of times a Sun position
              is inside each polygon in the Voronoi diagram.
            * **.vor_surf** (*``pyny.Surface``*): Voronoi diagram.
            * **.vor_centers** (*ndarray`*): Mass center of the 
              ``pyny.Polygons`` that form the Voronoi diagram.
        
        :param mesh_size: Mesh size for the square discretization of the
            Solar Horizont.
        :type mesh_size: float (in radians)
        :param plot: If True, generates a visualization of the Voronoi
            diagram.
        :type plot: bool
        :returns: None
        
        .. note:: In future versions this discretization will be
            improved substantially. For now, it is quite rigid and only
            admits square discretization.
        """
        from scipy.spatial import Voronoi
        from pyny3d.utils import sort_numpy
        state = pyny.Polygon.verify
        pyny.Polygon.verify = False

        # Sort and remove NaNs
        xy_sorted, order_back = sort_numpy(self.azimuth_zenit, col=1, 
                                           order_back=True)
        
        # New grid
        x1 = np.arange(-np.pi, np.pi, mesh_size)
        y1 = np.arange(-mesh_size*2, np.pi/2+mesh_size*2, mesh_size)
        x1, y1 = np.meshgrid(x1, y1)
        centers = np.array([x1.ravel(), y1.ravel()]).T
        
        # Voronoi
        vor = Voronoi(centers)
        
        # Setting the SH polygons
        pyny_polygons = [pyny.Polygon(vor.vertices[v], False)
                         for v in vor.regions[1:] if len(v) > 3]
        raw_surf = pyny.Surface(pyny_polygons)
                                 
        # Classify data into the polygons discretization
        map_ = raw_surf.classify(xy_sorted, edge=True, col=1, 
                                 already_sorted=True)
        map_ = map_[order_back]
        
        # Selecting polygons with points inside
        vor = []
        count = []
        for i, poly_i in enumerate(np.unique(map_)[1:]):
            vor.append(raw_surf[poly_i])
            bool_0 = map_==poly_i
            count.append(bool_0.sum())
            map_[bool_0] = i
                
        # Storing the information
        self.t2vor_map = map_
        self.vor_freq = np.array(count)
        self.vor_surf = pyny.Surface(vor)
        self.vor_centers = np.array([poly.get_centroid()[:2] 
                                     for poly in self.vor_surf])
        pyny.Polygon.verify = state

    def get_sunpos(self, t, true_time=False):
        """
        Computes the Sun positions for the *t* time vector.
        
        *t* have to be in absolute minutes (0 at 00:00 01 Jan). The and 
        in Sun positions calculated are in solar time, that is, maximun 
        solar zenit exactly at midday.
        
        The generated information is stored in:
            * **.azimuth_zenit** (*ndarray*)
            * **.true_time** (*datetime*): local time
        
        :param t: Absolute minutes vector.
        :type t: ndarray (dtype=int)
        :param true_time: If True, a datetime vector with the true local
            time will be stored at ``.true_time``
        :type true_time: bool
        :returns: Equivalent times in absolute minutes in year.
        :rtype: ndarray (dtype=int)
        
        :returns: None
        
        .. seealso:: :func:`to_minutes` to easily genetare valid input 
            t.
        """
        import numpy as np
        lat = self.arg_latitude
        long = self.arg_longitude
        alphamin = self.arg_zenitmin
        
        # Solar calculations
        day = np.modf(t/1440)[0]
        fractional_year = 2*np.pi/(365*24*60)*(-24*60+t)
        declination = 0.006918 - \
                      0.399912*np.cos(fractional_year) + \
                      0.070257*np.sin(fractional_year) - \
                      0.006758*np.cos(2*fractional_year) + \
                      0.000907*np.sin(2*fractional_year) - \
                      0.002697*np.cos(3*fractional_year) + \
                      0.00148*np.sin(3*fractional_year)
        
        hour_angle = np.tile(np.arange(-np.pi, np.pi, 2*np.pi/(24*60), 
                                       dtype='float'), 365)[t]
        solar_zenit = np.arcsin(np.sin(lat)*np.sin(declination) + \
                      np.cos(lat)*np.cos(declination)*np.cos(hour_angle))
        solar_zenit[solar_zenit<=0+alphamin] = np.nan
        #### Avoiding numpy warning
        aux = (np.sin(solar_zenit)*np.sin(lat) - np.sin(declination))/ \
              (np.cos(solar_zenit)*np.cos(lat))
        not_nan = np.logical_not(np.isnan(aux))
        aux_1 = aux[not_nan]
        aux_1[aux_1>=1] = np.nan
        aux[not_nan] = aux_1
        ####
        solar_azimuth = np.arccos(aux)
        solar_azimuth[day==0.5] = 0
        solar_azimuth[day<0.5] *= -1
        self.azimuth_zenit = np.vstack((solar_azimuth, solar_zenit)).T
                
        # True time
        if true_time:
            import datetime as dt
            long = np.rad2deg(long)
            instant_0 = dt.datetime(1,1,1,0,0,0) # Simulator time 
            
            # Real time
            equation_time = 229.18*(0.000075+0.001868*np.cos(fractional_year) - \
                            0.032077*np.sin(fractional_year) - \
                            0.014615*np.cos(2*fractional_year) - \
                            0.040849*np.sin(2*fractional_year))
        
            time_offset = equation_time + 4*long + 60*self.arg_UTC
            true_solar_time = t + time_offset
            delta_true_date_objs = np.array([dt.timedelta(minutes=i) 
                                             for i in true_solar_time])
            self.true_time = instant_0 + delta_true_date_objs
        
    def compute_shadows(self):
        """
        Computes the shadoing for the ``pyny.Space`` stored in 
        ``.space`` for the time intervals and Sun positions stored in 
        ``.arg_t`` and ``.sun_pos``, respectively.
        
        The generated information is stored in:
            * **.light_vor** (*ndarray (dtype=bool)*): Array with the 
              points in ``pyny.Space`` as columns and the discretized 
              Sun positions as rows. Indicates whether the points are 
              illuminated in each Sun position.
            * **.light** (*ndarray (dtype=bool)*): The same as 
              ``.light_vor`` but with the time intervals in ``.arg_t``
              as rows instead of the Sun positions.
        
        :returns: None
        """
        from pyny3d.utils import sort_numpy, bool2index, index2bool
        state = pyny.Polygon.verify
        pyny.Polygon.verify = False

        model = self.space
        
        light = []
        for sun in self.vor_centers:
            # Rotation of the whole ``pyny.Space``
            polygons_photo, _, points_to_eval = model.photo(sun, False)
            # Auxiliar pyny.Surface to fast management of pip
            Photo_surface = pyny.Surface(polygons_photo)
            Photo_surface.lock()
           
            # Sort/unsort points
            n_points = points_to_eval.shape[0]
            points_index_0 = np.arange(n_points) # _N indicates the depth level
            points_to_eval, order_back = sort_numpy(points_to_eval, col=0, 
                                                    order_back=True)
            # Loop over the sorted (areas) Polygons
            for i in model.sorted_areas:
                p = points_to_eval[points_index_0][:, :2]
                polygon_photo = Photo_surface[i]
                index_1 = bool2index(polygon_photo.pip(p, sorted_col=0))
                points_1 = points_to_eval[points_index_0[index_1]]
                
                if points_1.shape[0] != 0:
                    # Rotation algebra
                    a, b, c = polygon_photo[:3, :]
                    R = np.array([b-a, c-a, np.cross(b-a,  c-a)]).T
                    R_inv = np.linalg.inv(R)
                    Tr = a # Translation
                    # Reference point (between the Sun and the polygon)
                    reference_point = np.mean((a, b, c), axis=0)
                    reference_point[2] = reference_point[2] - 1
                    points_1 = np.vstack((points_1, reference_point))
                    points_over_polygon = np.dot(R_inv, (points_1-Tr).T).T
                    # Logical stuff
                    shadow_bool_2 = np.sign(points_over_polygon[:-1, 2]) != \
                                    np.sign(points_over_polygon[-1, 2])     
                    shadow_index_2 = bool2index(shadow_bool_2)
                    if shadow_index_2.shape[0] != 0:
                        points_to_remove = index_1[shadow_index_2]
                        points_index_0 = np.delete(points_index_0, 
                                                   points_to_remove)
            
            lighted_bool_0 = index2bool(points_index_0, 
                                        length=points_to_eval.shape[0])
            # Updating the solution
            light.append(lighted_bool_0[order_back])
        # Storing the solution
        self.light_vor = np.vstack(light)
        self.light = self.light_vor[self.t2vor_map]
        pyny.Polygon.verify = state

    def project_data(self):
        '''
        Assign the sum of ``.integral``\* to each sensible point in the
        ``pyny.Space`` for the intervals that the points are visible to 
        the Sun.

        The generated information is stored in:
            * **.proj_vor** (*ndarray*): ``.integral`` projected to the 
                Voronoi diagram.
            * **.proj_points** (*ndarray*): ``.integral`` projected to 
                the sensible points in the ``pyny.Space``.

        :returns: None
        
        .. note:: \* Trapezoidal data (``.arg_data``) integration over
            time (``.arg_t``).
        '''
        from pyny3d.utils import sort_numpy
        proj = self.light_vor.astype(float)
        
        map_ = np.vstack((self.t2vor_map, self.integral)).T
        map_sorted = sort_numpy(map_)
        
        n_points = map_sorted.shape[0]
        for i in range(proj.shape[0]):
            a, b = np.searchsorted(map_sorted[:, 0], (i, i+1))           
            if b == n_points:
                b = -1
            proj[i, :] *= np.sum(map_sorted[a:b, 1])
        self.proj_vor = np.sum(proj, axis=1)
        self.proj_points = np.sum(proj, axis=0)
        
    @staticmethod
    def to_minutes(time_obj = None, dt = None):
        '''
        Converts ``datetime`` objects lists into absolute minutes 
        vectors. It also can be used to generate absolute minutes vector
        from a time interval (in minutes).
        
        :param time_obj: ``datetime`` objects to convert into absolute 
            minutes.
        :type time_obj: list of ``datetime`` objects
        :param dt: Constant interval time to generate a time vector for
            a whole year.
        :type dt: int
        :returns: Equivalent times in absolute minutes in year.
        :rtype: ndarray (dtype=int)
        
        .. note:: If the time_obj has times higher than 23:59 31 Dec, 
            they will be removed.
        .. note:: If a leap-year is introduced, the method will remove
            the last year (31 Dec) in order to keep the series 
            continuous.
        '''
        import datetime

        if dt is not None and time_obj is None:
            return np.arange(0, 365*24*60, dt, dtype = int)
            
        elif dt is None and time_obj is not None:
            if type(time_obj) == datetime.datetime: 
                time_obj = [time_obj]
            
            year = time_obj[0].year
            time = []
            for obj in time_obj:
                tt = obj.timetuple()
                if year == tt.tm_year:
                    time.append((tt.tm_yday-1)*24*60 +
                                 tt.tm_hour*60 +
                                 tt.tm_min)
            return np.array(time, dtype=int)
            
        else:
            raise ValueError('Input error')
        
    
class Viz(object):
    '''
    This class stores the visualization methods. It is linked with 
    the ShadowsManager class by its attribute ``.viz``.
    
    :param ShadowsMaganer: ShadowsMaganer instance to compute the 
        visualizations.
    :returns: None
    '''
    def __init__(self, ShadowsMaganer):
        self.SM = ShadowsMaganer
        
    def vor_plot(self, which='vor'):
        """
        Voronoi diagram visualizations. There are three types:
        
            1. **vor**: Voronoi diagram of the Solar Horizont.
            2. **freq**: Frequency of Sun positions in t in the Voronoi
                diagram of the Solar Horizont.
            3. **data**: Accumulated time integral of the data projected 
                in the Voronoi diagram of the Solar Horizont.
        
        :param which: Type of visualization.
        :type which: str
        :returns: None
        """
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        sm = self.SM
        if sm.light_vor is None:
            raise ValueError('The computation has not been made yet')

        if which is 'vor':
            title = 'Voronoi diagram of the Solar Horizont'
            ax = sm.vor_surf.plot2d('b', alpha=0.15, ret=True, title=title)
            ax.scatter(sm.azimuth_zenit[:, 0],sm.azimuth_zenit[:, 1], c='k')
            ax.scatter(sm.vor_centers[:, 0], sm.vor_centers[:,1],
                        s = 30, c = 'red')
            ax.set_xlabel('Solar Azimuth')
            ax.set_ylabel('Solar Zenit')
            plt.show()

        elif which is 'freq':
            cmap = cm.Blues
            title = 'Frequency of Sun positions in the Voronoi diagram '+\
                     'of the Solar Horizont'
            ax = sm.vor_surf.plot2d(sm.vor_freq, cmap=cmap, alpha=0.85, 
                                    colorbar=True, title=title, ret=True,
                                    cbar_label='    Freq')
            ax.set_xlabel('Solar Azimuth')
            ax.set_ylabel('Solar Zenit')
            plt.show()
            
        elif which is 'data':
            cmap = cm.YlOrRd
            title = 'Data projected in the Voronoi diagram of the'+\
                    ' Solar Horizont'
            data = sm.proj_vor/sm.vor_freq
            proj_data = data*100/data.max()
            ax = sm.vor_surf.plot2d(proj_data, alpha=0.85, cmap=cmap,
                                    colorbar=True, title=title, ret=True,
                                    cbar_label='%')
            ax.set_xlabel('Solar Azimuth')
            ax.set_ylabel('Solar Zenit')
            plt.title('max = '+str(data.max())+' kilounits*hour')
            plt.show()
            
        else:
            raise ValueError('Invalid plot '+which)
        
    def exposure_plot(self, places=-1, c_poly='default', c_holes='default',
                      s_sop=25, extra_height=0.1):
        """
        Plots the exposure of the sensible points in a space to the data
        and the Sun positions. It is required to previously compute the 
        shadowing.
        
        If the computation has been made with a data timeseries, the plot
        will have a colorbar. Units are accumulated kilounits*hour (for 
        the series), that is, if the input data is in Watts 
        (irradiation) for a whole year, the output will be 
        kWh received in an entire year.
        
        If there is no data inputed, the plot will show only the number
        of times each point "has been seen by the Sun" along the series.
        
        :param places: Indexes of the places to plot. If -1, plots all.
        :type places: int or list
        :param c_poly: Polygons color.
        :type c_poly: matplotlib color, 'default' or 't' (transparent)
        :param c_holes: Holes color.
        :type c_holes: matplotlib color, 'default' or 't' (transparent)
        :param s_sop: Set of points size.
        :type s_sop: float or ndarray
        :param extra_height: Extra elevation for the points in the plot.
        :type extra_height: float
        :returns: None
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        sm = self.SM
        if sm.light_vor is None:
            raise ValueError('The shadowing has not been computed yet')
        proj_data = sm.proj_points*100/sm.proj_points.max()
        
        if places == -1: 
            places = range(len(sm.space.places))
        elif type(places) == int:
            places = [places]
            
        places = np.array(places)
        places[places<0] = len(sm.space.places) + places[places<0]
        places = np.unique(places)
        
        points = sm.space.get_sets_of_points()        
        index = sm.space.get_sets_index()        
        
        # Model plot
        sop = []
        data = []
        aux_space = pyny.Space() # Later centering of the plot
        ax=None
        for i in places:
            aux_space.add_places(sm.space[i])
            ax = sm.space[i].iplot(c_poly=c_poly, c_holes=c_holes, 
                                   c_sop=False, ret=True, ax=ax)
            sop.append(points[index==i])
            data.append(proj_data[index==i])
        sop = np.vstack(sop)
        sop = np.vstack((sop, np.array([-1e+12, -1e+12, -1e+12])))
        data = np.hstack(data)
        proj_data = np.hstack((data, 0))
                              
        # Sensible points plot
        ## Color
        cmap = cm.jet
        normalize = mcolors.Normalize(vmin=proj_data.min(), 
                                      vmax=proj_data.max())
        color_vector = cmap(normalize(proj_data))
        ## Plot
        ax.scatter(sop[:, 0], sop[:, 1], sop[:, 2]+extra_height, 
                   c=color_vector, s=s_sop)
        ## Axis
        aux_space.center_plot(ax)
                   
        ## Colorbar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(proj_data)
        cbar = plt.colorbar(scalarmappaple, shrink=0.8, aspect=10)
        cbar.ax.set_ylabel('%', rotation=0)
        if not (sm.arg_data.max() == 1 and sm.arg_data.min() == 1):
            plt.title('Accumulated data Projection\nmax = ' + \
                      str(sm.proj_points.max()) + \
                      ' kilounits*hour')
        else:
            plt.title('Sun exposure')
        