Back to :ref:`tutorials`

.. contents::
    :local:
    
.. _tutorial_basic_usage:

Classes usage
=============
In this guide we are going to use the geometry created in the tutorial 1 
(:ref:`tutorial_building_by_aggregation`). As we saw there, the most direct
way to initinialize it is::

    import numpy as np
    import pyny3d.geoms as pyny

    # Declaring the geometry
    ## Surface
    poly_surf_0 = [np.array([[0,0,0], [7,0,0], [7,10,2], [0,10,2]]),
                   np.array([[0,10,2], [7,10,2], [3,15,3.5]]),
                   np.array([[0,10,2], [3,15,3.5], [0,15,3.5]]),
                   np.array([[7,10,2], [15,10,2], [15,15,3.5], [3,15,3.5]])]
    poly_surf_1 = [np.array([[8,0,0], [15,0,0], [15,9,0], [8,9,0]])]

    ## Obstacles
    wall_1 = np.array([[0,0,4], [0.25,0,4], [0.25,15,4], [0,15,4]])
    wall_2 = np.array([[0,14.7,5], [15,14.7,5], [15,15,5], [0,15,5]])
    chimney = np.array([[4,0,7], [7,0,5], [7,3,5], [4,3,7]])

    # Building the solution
    place_0 = pyny.Place(poly_surf_0, melt=True)
    place_0.add_extruded_obstacles([wall_1, wall_2, chimney])
    place_1 = pyny.Place(poly_surf_1)
    space = pyny.Space([place_0, place_1])
    space.mesh(0.5)

To check that everything is correct we can visualize the whole Space::

    space.iplot(c_poly='b')

.. figure:: ../images/tutorials/1_building_by_aggregation/space_1.png
   :scale: 60%
   :align: center

Iterability and indexability
----------------------------
An important issue to consider is that all classes but Place are iterable and
indexable:

    ==================  ==================
          class           indexes its...
    ==================  ==================
    Polygon             ... points
    Surface             ... Polygons
    Polyhedron          ... Polygons
    Space               ... Places
    ==================  ==================

The Place class is not indexable because it is formed by an heterogeneous
mix of objects (Surface, Polyhedra and Points). On the other hand, the holes in
a Surface are stored in the attribute ``Surface.holes``.

Thanks to this it is quite easy accessing to any object::

    # Single step calls
    ## Extracting from a Space
    place = space[0]
    ## Extracting from a Place
    surface = place.surface
    polyhedra = place.polyhedra
    set_of_points = place.set_of_points
    ## Extracting from a Surface
    surf_poly = surface[1]
    ## Extracting from a Polyhedron
    polyhedron = polyhedra[3]   # Remember there are multiple obstacles
    polyhedron_poly = polyhedron[2]
    ## Extracting from a Polygon
    point = polyhedron_poly[-1]
    
    # Multiple steps calls (clean)
    point = space[0].polyhedra[3][2][-1]
    
    # Alternative way (verbose)
    point = space.places[0].polyhedra[3].polygons[2].points[-1]

The three ``point`` above are exactly the same: ``array([7, 3, 0.6])``. You can
think that the *clean* version is not the clearest. And you are probably right.
The key here is to be comfortable with one type of indexing.

In my personal case, if I have doubts, I always take a look to the scheme at 
the top of the :ref:`scheme` section and think of it backwards: *A Space holds 
Places. A Place holds Polyhedra, Points and a Surface. Both, Polyhedron and 
Surface hold Polygons. And, a Polygon holds Points.*
    
The only thing that you have to remember is that ``place.polyhedra`` is a list.
If you want to take just one Polyhedron, you have to index it just like we did
before::

    polyhedron = polyhedra[3]   # Remember there are multiple obstacles
    
Also, this structure makes easy working with loops (specially list 
comprehensions) and, in general, makes the code cleaner. Here we can see some 
examples in an IPython interactive session:

.. ipython::
    :verbatim:
    
    In [1]: # Getting the parametric equations of all faces in a Polyhedron
       ...: eq = [face.get_parametric() for face in polyhedron]
       ...: eq
    Out[1]: 
    [array([-21, 0, 0, 84]),
     array([0, 21, 0, 0]),
     array([-15, 0,  0, 105]),
     array([0, 19.2, 0, -57.6]),
     array([0,  1.8, -9, 0]),
     array([-6, 0, -9, 87])]
        
    In [2]: # Getting the area of all surfaces in a space
       ...: areas = [place.surface.get_area() for place in space]
       ...: areas
    Out[2]: [131.95765088324114, 63.0]  

    In [3]: # Getting the shapely.Polygon object of all holes in a Surface
       ...: holes_shapely = [hole.get_shapely() for hole in surface.holes]
       ...: holes_shapely
    Out[3]: 
    [<shapely.geometry.polygon.Polygon at 0x53d2cd78d0>,
     <shapely.geometry.polygon.Polygon at 0x53d2cd71d0>,
     <shapely.geometry.polygon.Polygon at 0x53d2cd7208>,
     <shapely.geometry.polygon.Polygon at 0x53d2cd7128>]

Seeds
-----
In *pyny3d* a seed is dict/list structure which can be used to clone an 
instance of a class by giving the enough information in a \*\**kwargs* form.
Every class can generate seeds.

Seeds are extremely useful because they contain well-structured data of the
geometries. As methods gain complexity they use seeds more and more. Even, if
you are going to write some functions to control and/or generate information
from *pyny3d* objects you will want to know what seeds are.

.. ipython::
    :verbatim:
    
    In [4]: # Polygon
       ...: surf_poly.get_seed()    
    Out[4]: 
    {'points': array([[0, 10, 2],
                      [15, 10, 2],
                      [15, 15, 3.5],
                      [0, 15, 3.5]])}

    In [5]: # Place
        ...: place.get_seed()
    Out[5]: 
    {'polyhedra': [[array([[0, 10, 4],
                           [0, 0, 4],
                           [0, 0, 0],
                           [0, 10, 2]]), 
                    array([[0.25, 0, 4],
                           [0, 0, 4],
                           [0, 0, 0],
                           [0.25, 0, 0]]), 
                    array([[0.25, 10, 4],
                           [0.25, 0, 4],
                           [0.25, 0, 0],
                           [0.25, 10, 2]]), 
                    array([[0.25, 10, 4],
                           [0, 10, 4],
                           [0, 10, 2],
                           [0.25, 10, 2]]), 
                    array([[0, 0, 0],
                           [0.25, 0, 0],
                           [0.25, 10, 2],
                           [0, 10, 2]]), 
                    array([[0, 0, 4],
                           [0.25, 0, 4],
                           [0.25, 10, 4],
                           [0, 10, 4]])], 
                   [array([[0, 15, 4],
                           [0, 10, 4],
                           [0, 10, 2],
                           [0, 15, 3.5]]), 
                    array([[0.25, 10, 4],
                           [0, 10, 4],
                           [0, 10, 2],
                           [0.25, 10, 2]]), 
                    array([[0.25, 15, 4],
                           [0.25, 10, 4],
                           [0.25, 10, 2],
                           [0.25, 15, 3.5]]), 
                    array([[0.25, 15, 4],
                           [0, 15, 4],
                           [0, 15, 3.5],
                           [0.25, 15, 3.5]]), 
                    array([[0, 10, 2],
                           [0.25, 10, 2],
                           [0.25, 15, 3.5],
                           [0, 15, 3.5]]), 
                    array([[0, 10, 4],
                           [0.25, 10, 4],
                           [0.25, 15, 4],
                           [0, 15, 4]])], 
                   [array([[0, 15, 5],
                           [0, 14.7 , 5],
                           [0, 14.7 , 3.41],
                           [0, 15, 3.5]]), 
                    array([[15, 14.7 , 5],
                           [0, 14.7 , 5],
                           [0, 14.7 , 3.41],
                           [15, 14.7 , 3.41]]), 
                    array([[15, 15, 5],
                           [15, 14.7 , 5],
                           [15, 14.7 , 3.41],
                           [15, 15, 3.5]]), 
                    array([[15, 15, 5],
                           [0, 15, 5],
                           [0, 15, 3.5],
                           [15, 15, 3.5]]), 
                    array([[0, 14.7 , 3.41],
                           [15, 14.7 , 3.41],
                           [15, 15, 3.5],
                           [0, 15, 3.5]]), 
                    array([[0, 14.7, 5],
                           [15, 14.7, 5],
                           [15, 15, 5],
                           [0, 15, 5]])], 
                   [array([[4, 3, 7],
                           [4, 0, 7],
                           [4, 0, 0],
                           [4, 3, 0.6]]), 
                    array([[7, 0, 5],
                           [4, 0, 7],
                           [4, 0., 0],
                           [7, 0., 0]]), 
                    array([[7, 3, 5],
                           [7, 0, 5],
                           [7, 0, 0],
                           [7, 3, 0.6]]), 
                    array([[7, 3, 5],
                           [4, 3, 7],
                           [4, 3, 0.6],
                           [7, 3, 0.6]]), 
                    array([[4, 0, 0],
                           [7, 0, 0],
                           [7, 3, 0.6],
                           [4, 3, 0.6]]), 
                    array([[4, 0, 7],
                           [7, 0, 5],
                           [7, 3, 5],
                           [4, 3, 7]])]],
     'set_of_points': array([[0.5, 0, 0.1],
                             [1, 0, 0.1],
                             [1.5, 0, 0.1],
                             ..., 
                             [14, 14.5, 3.45],
                             [14.5, 14.5, 3.45],
                             [15, 14.5, 3.45]]),
     'surface': {'holes': [array([[0, 0, 0],
                                  [0.25, 0, 0],
                                  [0.25, 10, 2],
                                  [0, 10, 2]]), 
                           array([[0, 10, 2],
                                  [0.25, 10, 2],
                                  [0.25, 15, 3.5],
                                  [0, 15, 3.5]]), 
                           array([[0, 14.7, 3.41],
                                  [15, 14.7, 3.41],
                                  [15, 15, 3.5],
                                  [0, 15, 3.5]]), 
                           array([[4, 0, 0],
                                  [7, 0, 0],
                                  [7, 3, 0.6],
                                  [4, 3, 0.6]])], 
                 'polygons': [array([[0, 0, 0],
                                     [7, 0, 0],
                                     [7, 10, 2],
                                     [0, 10, 2]]), 
                              array([[0, 10, 2],
                                     [15, 10, 2],
                                     [15, 15, 3.5],
                                     [0, 15, 3.5]])]}} 
              
As we can see in the last output, the seed contains all the information about
the Place's polyhedra, set_of_points and surface (distinguishing between 
polygons and holes in this). To clone an instance we can use the next syntax:

.. ipython::
    :verbatim:
    
    In [6]: seed = place.get_seed()
       ...: place_clone = pyny.Place(**seed)

Or you can directly use the method created for that:

.. ipython::
    :verbatim:
    
    In [7]: place_clone = place.seed2pyny(seed)

``place_clone`` are identical between them and completely equivalent to 
``place``.

Copy, Save and Restore
----------------------
Making use of *seeds*, ``.copy()``, ``.save()`` and ``.restore()`` global 
methods are available to manage all the classes in *pyny3d*. It is possible
to use them on any class at any time and anywhere.

.. ipython::
    :verbatim:
    
    In [8]: place_clone = place.copy()

``.save()`` and ``.restore()`` are used the following way:

.. ipython::
    :verbatim:
    
    In [9]: place.save()

    In [10]: # ... some modifications ... #

    In [11]: place = place.restore()

These global methods are extremely useful for interactive sessions where the
user is approaching a problem for the first time because they make possible to 
stop worrying about models' fragility. It is trivial to create as many
backup copies and *saves* as desired.

.. seealso:: :ref:`tutorial_space` 
    
    Space class have another kind of seed called *map*. It is exactly
    the same but all the information is arranged in a single *ndarray* what
    makes a lot easier performing transformations like traslations or 
    rotations.

Other *get* global methods
--------------------------
You will find several *getters* and some *setters* in *pyny3d*. They makes the
interaction between the user and the objects easier and clearer. Two global
*getters* are ``.get_domain()`` and ``.get_centroid()``, both widely used 
internally. Its usage is, again, trivial:

.. ipython::
    :verbatim:

    In [12]: place = place.restore()

    In [13]: place.get_domain()
    Out[13]: 
    array([[0, 0, 0],
           [15, 15, 7]])

    In [14]: place.get_centroid()
    Out[14]: array([7.5, 7.5, 3.5])

|

Next tutorial: :ref:`tutorial_visualizations`













