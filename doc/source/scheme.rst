.. contents::
    :local:

.. _scheme:
    
Modules scheme
==============
.. figure:: ../images/scheme.png
   :scale: 70%
   :align: center
   
   *Simple Class Diagram (classes in green)*

As you probably know, there are not public or private methods in Python, it is
considered that every single line in a Python library is accessible by 
everyone. It is an user decision choosing the most convenient depth of usage of
a certain package.

For this reason, I encourage you to use ``pyny3d`` the way you want, from a
very superficial usage of a couple of functions to a deep editing of a class.
This library has been made to be as simple as possible for reasons like this.

But indepently of your implication with the project, it would be a big mistake 
on my part not to advise you the most *usable or higher-level* functions.

Take in account that most this functions are used in the examples shown in the 
:ref:`tutorials` section, I encourage you to take a closer look there. For the 
current section it will be enough just listing them with a short description.

``geoms`` module
----------------

Common methods
~~~~~~~~~~~~~~

    ==================  =======================================================
          method                        description  
    ==================  =======================================================
    .move()             Translate the object
    .zrotate()          Rotates the object about a z axis
    .mirror()           Generates a symmetry of the object
    .matrix()           Copy the object along a 3D matrix
    .get_domain()       Returns the domain of an instance
    .plot()             Generates a 3D plot
    .save()             Save the current state of an instance
    .restore()          Restores a previous state of an instance
    .copy()             Copy the current state of an instance
    ==================  =======================================================
    
|

Polygon
~~~~~~~

    ==================  =======================================================
          method                       description  
    ==================  =======================================================
    .get_parametric()   Returns the parametric equation of the polygon's plane
    .get_path()         Returns the matplotlib.path.Path of the z=0 projection
    .get_shapely()      Returns the shapely.Polygon of the z=0 projection
    .get_area()         Returns the real area
    .get_height()       Returns the z value for the parametric equation for 
                        a list of points
    .is_convex()        Returns whether a polygon is convex
    .make_ccw()         Changes the order of a set of points to be ccw
    .to_2d()            Generates the real 2D polygon of the 3D polygon
    .contains()         Points-in-Polygon algorithm for the z=0 projection
    .pip()              Faster Points-in-Polygon algorithm
    .plot2d()           2D plot of the z=0 projection
    .lock()             Precomputes some values to speedup shadowing
    ==================  =======================================================

|

Surface
~~~~~~~

    ==================  =======================================================
          method                       description  
    ==================  =======================================================
    .get_area()         Returns the real area
    .get_height()       Returns the z value for a list of points
    .classify()         Points-in-Polygon for multiple non-overlapping polygons
    .interct_with()     Returns the intersection of multiple polygons
    .contiguous()       Returns whether a set of polygons are contiguous
    .melt()             Merge groups of coplanar and contiguous polygons
    .add_holes()        Add holes to the Surface
    .iplot()            3D plot with polygons/holes color control
    .plot2d()           2D plot of the z=0 projection
    .lock()             Precomputes some values to speedup shadowing
    ==================  =======================================================
    
|

Polyhedron
~~~~~~~~~~

    ==================  =======================================================
          method                       description  
    ==================  =======================================================
    .get_area()         Returns the real area
    .by2polygons()      Create a polyhedron connecting two polygons 
    ==================  =======================================================
    
|

Place
~~~~~

    ==========================      ===========================================
          method                                    description  
    ==========================      ===========================================
    .add_set_of_points()            Add a set of points
    .clear_set_of_points()          Remove the points in this place
    .add_extruded_obtacles()        Add a polyhedra connecting one polyhedron 
                                    with the surface
    .get_height()                   Returns the z value for a list of points
    .mesh()                         Generates a list of points homogeneously 
                                    distributed
    .add_holes()                    Add holes to the surface
    .iplot()                        3D plot with color and size control
    ==========================      ===========================================
    
|

Space
~~~~~

    ==========================      ===========================================
          method                                    description  
    ==========================      ===========================================
    .add_places()                   Add new places to the space
    .add_space()                    Merge other spaces with this one
    .add_set_of_points()            Add a set of points
    .clear_set_of_points()          Remove the points in this place
    .get_map()                      Returns all the points that forms 
                                    everything declared in the space
    .map2pyny()                     Creates a Space from a map
    .map2seed()                     Creates a Space from a seed
    .explode()                      Collect all the polygons, holes and points 
                                    in the space
    .explode_map()                  Faster version of .explode()
    .get_height()                   Returns the z value for a list of points
    .mesh()                         Generates a list of points homogeneously 
                                    distributed
    .photo()                        Change of the reference system to align the
                                    an axis with a given direction
    .iplot()                        3D plot with place, color and size control
    .shadows()                      Initializes the ShadowingManager
    .lock()                         Precomputes some values to speedup 
                                    shadowing
    ==========================      ===========================================


``shadows`` module
------------------

ShadowsManager
~~~~~~~~~~~~~~

    ==================  =======================================================
          method                       description  
    ==================  =======================================================
    .to_minutes()       Generates absolute minutes time series
    .get_sunpos()       Computes the Sun positions for a time series
    .Voronoi_SH()       Discretizes the Solar Horizont through a Voronoi 
                        diagram
    .compute_shadows()  Computes the shadowing
    .project_data()     Assign data time series to the places illuminated by 
                        the Sun
    ==================  =======================================================
    
|

Viz
~~~

    ==================  =======================================================
          method                       description  
    ==================  =======================================================
    .vor_plot()         Generates visualizations about the Voronoi 
                        discretization
    .exposure_plot()    Generates a 3D visualization with the projected data
                        from the Sun to the ``pyny.Space``
    ==================  =======================================================























