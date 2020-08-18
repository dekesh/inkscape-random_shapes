#! /usr/bin/python2
'''


This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
import inkex, simplestyle, math
from simpletransform import computePointInNode
import random
import copy
from numpy import cos,sin,pi
import numpy as np
#SVG element generation routine
def draw_SVG_square((w,h), (x,y), parent):

    style = {   'stroke'        : 'none',
                'stroke-width'  : '1',
                'fill'          : '#000000'
            }

    transform_str = 'rotate (' +str(random.random()*360) + ')'
    
    attribs = {
        'style'     : simplestyle.formatStyle(style),
        'height'    : str(h),
        'width'     : str(w),
        'transform' : transform_str,
        'x'         : str(x),
        'y'         : str(y)
            }
    circ = inkex.etree.SubElement(parent, inkex.addNS('rect','svg'), attribs )

class Random_shape(inkex.Effect):
    def __init__(self):
        inkex.Effect.__init__(self)
        self.OptionParser.add_option("-L", "--active_layer",
                        action="store", type="string",
                        dest="active_layer", default="",
                        help="Active layer to take pictures from")
        self.OptionParser.add_option("-r", "--max_rot",
                        action="store", type="float",
                        dest="max_rot", default=0,
                        help="Maximum rotation for any of the objects")
        self.OptionParser.add_option("-s", "--min_scale",
                        action="store", type="float",
                        dest="min_scale", default=0.5,
                        help="The minimum scaling for any of the objects")
        self.OptionParser.add_option("-N", "--num_rep",
                        action="store", type="int",
                        dest="num_rep", default=10,
                        help="Maximum number of duplicated objects")
        self.OptionParser.add_option("-o", "--overlap",
                        action="store", type="inkbool",
                        dest="allow_overlap", default=False,
                        help="Allow overlap of duplicated objects")
        self.OptionParser.add_option("--inv_x",
                        action="store", type="inkbool",
                        dest="inv_x", default=False,
                        help="Randomize flip-x")
        self.OptionParser.add_option("--inv_y",
                        action="store", type="inkbool",
                        dest="inv_y", default=False,
                        help="Randomize flip-y")
        self.OptionParser.add_option("--scale_decrease",
                        action="store", type="inkbool",
                        dest="scale_decrease",
                        help="Decrease maximum scale during placement of new images")
        self.OptionParser.add_option("--strict_borders",
                        action="store", type="inkbool",
                        dest="strict_borders",
                        help="Do not allow images to be placed out of page borders")
        self.OptionParser.add_option("--tab",
                        action="store", type="string",
                        dest="tab",
                        help="The selected UI-tab when OK was pressed")
        
        

    def effect(self):

        # Select required layer to take elements from.
        layer_name = self.options.active_layer
        if layer_name:
            figs_layer = self.document.xpath('//svg:g[@inkscape:label="' + layer_name + '"]', namespaces=inkex.NSS)
            if not(figs_layer):
                inkex.errormsg("There isn't a layer named: " + layer_name + ". I'm quitting")
                return None
            figs_layer = figs_layer[0]
        else:
            figs_layer = self.current_layer

        fig_list = self.figure_list(figs_layer)
        if not(fig_list):
            # Note that layer does not necessary has a label
            if inkex.addNS('label','inkscape')  in  figs_layer.attrib:
                errstr = "I couldn't find any figures ('image' tag) in layer: '" + figs_layer.attrib[ inkex.addNS('label','inkscape')] +"'." 
            else:
                errstr = "I couldn't find any figures ('image' tag) in chosen layer."
            inkex.errormsg(errstr + " Note that images may be in a sublayer of chosen layer. I'm quitting")
            return None

        # New layer to draw upon
        new_layer = self.new_layer(figs_layer)       

        self.draw_figs(new_layer, fig_list)
        

    def figure_list(self,parent_layer, tags=['image']):
        figs = parent_layer.findall('svg:image', inkex.NSS)
        o = []
        # drop elements without x,y height and width. We don't know what to do with elements like that

        for i in figs:
            try:
                _,_,_,_ = (i.attrib['x'],i.attrib['y'],i.attrib['width'],i.attrib['height'])  #         attrib_list = ['x','y','width','height']
                o.append(i)
            except KeyError as e:
                pass
        return o

    def new_layer(self, figs_layer):
        """
        Create new layer
        """
        svg = figs_layer.getparent()
        # inkex.errormsg(str(figs_layer.attrib))
        new_name = "(output)"
        # Note that layer does not necessary has a label

        if inkex.addNS('label','inkscape') in figs_layer.attrib:
            new_name = figs_layer.attrib[ inkex.addNS('label','inkscape') ] + new_name
        else:
            new_

        new_id = self.uniqueId(figs_layer.attrib['id'])
        layer_attribs = {
            inkex.addNS('groupmode','inkscape'):"layer",
            inkex.addNS('label','inkscape'):new_name,
            'id':new_id,
            'style':'display:inline'}
        new_layer = inkex.etree.SubElement(svg, 'g', layer_attribs)
        return new_layer

    def draw_figs(self, layer, fig_list):
        """
        Put images in fig_list on layer in a random order.
        Settings are determined through self.options.#
        """
        # We are working in document units of length.
        min_y = 0; max_y = float(self.getDocumentHeight()[0:-2])
        min_x = 0; max_x = float(self.getDocumentWidth()[0:-2])

        figs_edge_list = []  # for each figure added contains the edges in the form [A1,A2,A3,A4] with Ai a tuple (x,y) points.
        num_figs = 0
        num_tries = 0
        max_tries = 4000        # maximum number of attempts to put figures on layer. 
        max_scale = 1
        inv_x = 1               # 1 for not inverted. -1 for inverted
        inv_y = 1
        d = {}                  # holds figure properties. position, rotation, scale

        while (num_tries < max_tries and num_figs < self.options.num_rep):
            
            # reduce scale after few non-successful events
            if self.options.scale_decrease and not(num_tries%10): 
                max_scale = self.options.min_scale +  0.8 * (max_scale - self.options.min_scale) 

            # Pick a random figure and set random size, position, rotation, flip
            fig = random.choice(fig_list)            
            scale = random.uniform( self.options.min_scale, max_scale)
            # width =  float( self.unittouu(fig.attrib['width']))*scale
            # height = float( self.unittouu(fig.attrib['width']))*scale
            try:
                width = float(fig.attrib['width'])*scale
            except:
                width = float(self.unittouu(fig.attrib['width']))*scale

            try:
                height = float(fig.attrib['height'])*scale
            except:
                height = float( self.unittouu(fig.attrib['height']))*scale
                
            x = random.random()*max_x
            y = random.random()*max_y
            rotation = (2 - 4*random.random())*random.random()*self.options.max_rot
            if self.options.inv_x:
                inv_x = random.choice([-1,1])
            if self.options.inv_y:
                inv_y = random.choice([-1,1])

            # Set SVG properties for image
            d['width'] =  str(width)
            d['height'] = str(height)
            d['x'] = str(0)
            d['y'] = str(0)
            scale_str = ' scale(' + str(inv_x) + ',' + str(inv_y) + ')'
            transform_str = 'translate(' + str(x) + ',' + str(y) + ') ' + 'rotate('+  str(rotation) + ')'  + scale_str
            d['transform'] = transform_str

            # inkex.errormsg('width:' + d['width'] + "   scale: " + str(scale) + '  width: ' + fig.attrib['width'] )
            
            # Return position of the four edges of the new image
            pointsA = math_helper.obtain_edge_rectangle(x,y,width,height,rotation, inv_x, inv_y)

            # Check if figure intersect or contained in other figure. If figure completely covers other figure - we just ignore it,
            # since the latter will not be seen in the final result
            intersection = False
            if self.options.allow_overlap:
                intersection = False
            else:
                for pointsB in figs_edge_list:
                    if math_helper.do_shapes_intersect(pointsA,pointsB):
                        intersection = True
                        break
                    if math_helper.is_point_contained_in_shape(pointsA[0], pointsB):
                        intersection = True
                        break

            # Check if figure protrude out of border
            out_of_borders = False
            if self.options.strict_borders:
                for pt in pointsA:
                    if pt[0] < 0 or pt[0] > max_x or pt[1] < 0 or pt[1] > max_y:
                        out_of_borders = True
                        break

            if intersection or out_of_borders :
                num_tries += 1
            else:
                figs_edge_list.append(pointsA)
                self.add_figure(layer, d, fig)
                num_figs += 1

    def add_figure(self, layer, new_attribs, fig):
        new_fig = copy.deepcopy(fig)
        new_fig.attrib['id'] = self.uniqueId(fig.attrib['id'])
        new_fig.attrib.update(new_attribs)
        layer.append(new_fig)
        


class math_helper:
    """
    Some helper functions
    """

    @staticmethod
    def obtain_edge_rectangle(x,y,width,height,rotation, inv_x, inv_y):
        """
        returns the  four edges of a rectangle in a list of four tuples (x,y). The rectangle is first positioned so that it's "lowest" corner is in (x,y)
        then rotated about its (x,y) corner (clockwise?), and lastly it may be inverted through the x- or y-axis
        """
        rot = (2*pi/360)*rotation
        rot_mat = np.array([[cos(rot), -sin(rot)],[sin(rot),cos(rot)]])
        zero = np.array([x,y])
        v1 = np.matmul(rot_mat, (0,height))  # vector that points to the left-top corner
        v2 = np.matmul(rot_mat, (width,0))  # vector that points to the right-bottom corner
        a = zero + v1
        b = zero + v1 + v2
        c = zero + v2
        if inv_y == -1:
            a -= 2*v1
            b -= 2*v1
        if inv_x == -1:
            b -=2*v2
            c -=2*v2
        
        return [(x,y),tuple(a), tuple(b), tuple(c)]
        

    @staticmethod
    def is_point_contained_in_shape(point, pointsA):
        """
        Returns true if point = (x,y) is contained in the shape defined by the tuple list pointsA.
        """

        # If the maximal distance between the point and the edges is greater than the maximal distance between
        # all of the edges we assume point is not contained in shape (this is wrong, since point can be out of figure
        # and be considered as inside, but it still works fine enough. There is probably some canonical way to do it..)
        max_dist_point = 0
        pt = np.array(point)
        for i in pointsA:
            max_dist_point = max(np.linalg.norm(pt-i),max_dist_point)

        max_dist_shape = 0
        for i in pointsA:
            for j in pointsA:
                max_dist_shape = max(np.linalg.norm(np.array(j)-i),max_dist_shape)
        if max_dist_shape > max_dist_point:
            return True
                
    
    @staticmethod
    def do_shapes_intersect(pointsA, pointsB):
        """
        Find out whether two shapes partially overlap. The shapes are composed of a list of points connected by line (last and first points connected).
        Returns True if shapes overlap.
        pointsA, pointsB:  A list of tuples (x,y) which form the edges of the shapes.
        """

        # Check if any of the segements intersect
        for i in range(len(pointsA)):
            for j in range(len(pointsB)):
                if math_helper.do_segment_intersect(pointsA[i-1], pointsA[i], pointsB[j-1], pointsB[j]) :
                    return True

        return False
                
    @staticmethod
    def do_segment_intersect(A1,A2,B1,B2):
        """ 
        Returns True if the two segments A and B defined by the endpoints A1 and A2, and B1 and B2 intersect.
        Ai and Bi are tuples (x,y)
        """
        A1x,A1y = A1; A2x, A2y = A2
        B1x,B1y = B1; B2x, B2y = B2

        #x_inter, y_inter - The solution to the point of intersection of two (infinite) lines
        x1, y1 = (A1x - A2x), (A1y - A2y)
        x2, y2 = (B1x - B2x), (B1y - B2y)
        d = x2*y1 - x1*y2
        eps = 0.000001
        if abs(d)<eps:          # if lines meet at infinity, segments do not intersect.
            return False
        x_inter = 1.*((B1y-A1y)*x1*x2  + A1x*x2*y1 - B1x*x1*y2)/d
        y_inter = 1.*(-B1y*x2*y1 + A1y*x1*y2 + (B1x - A1x)*y1*y2 )/(-d)

        # Check if intersection point occurs on the given segments
        inter = np.array([x_inter,y_inter])
        A1n = np.array(A1); A2n = np.array(A2)
        B1n = np.array(B1); B2n = np.array(B2)
        if (A1n-inter).dot((A1n-inter)) + (A2n-inter).dot((A2n-inter)) < (A1n-A2n).dot(A1n-A2n):  # Inside segment A
            if (B1n-inter).dot((B1n-inter)) + (B2n-inter).dot((B2n-inter)) < (B1n-B2n).dot(B1n-B2n):  # Inside segment B
                return True
        
        return False
if __name__ == '__main__':
    e = Random_shape()
    e.affect()

