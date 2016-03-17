#ifdef GL_ES
    precision mediump float;
#endif
varying vec3 vDirection;
uniform float eye;
uniform float projection;
uniform float time;


uniform sampler2D uSampler;

#define PI 3.1415926535897932384626433832795
const float pi = 3.14159;
const float pi2 = 2.0*pi;

/* This should not be strictly necessary; the python code was designed to work in absolute pixel coordinates,
    but the video player uses UV coordinates. We end up converting back and forth for no good reason right now. 
*/
const float WIDTH = 1028.0; 
const float HEIGHT = WIDTH /2.0;

vec2 iMouse = vec2(WIDTH/2.,HEIGHT/2.);
float iGlobalTime = 10.0;
float determinant(mat2 m) {
return m[0][0]*m[1][1]-m[0][1]*m[1][0];
}

mat2 inverse(mat2 m) {
    return mat2(m[1][1], -m[0][1], -m[1][0], m[0][0]) / determinant(m);
}

//https://github.com/mkovacs/reim/blob/master/reim.glsl

vec2 cConj(vec2 c)
{
    return vec2(c.x, -c.y);
}
float cReal(vec2 c)
{
    return c.x;
}
vec2 complex(float r, float i){
    return vec2(r,i);
}
float cImag(vec2 c)
{
    return c.y;
}
vec2 cNeg(vec2 c)
{
    return -c;
}
vec2 cInv(vec2 c)
{
    return cConj(c) / dot(c, c);
}
vec2 cMul(vec2 a, vec2 b)
{
    return vec2(a.x*b.x - a.y*b.y,
                a.x*b.y + a.y*b.x);
}
vec2 cDiv(vec2 a, vec2 b)
{
    return cMul(a, cInv(b));
}

vec2 cCis(float r)
{
  return vec2( cos(r), sin(r) );
}

float cArg(vec2 c)
{
  return atan(c.y, c.x);
}
float cAbs(vec2 c)
{
  return length(c);
}
vec2 cCish(float r)
{
  vec2 e = vec2( exp(r), exp(-r) );
  return vec2(e.x + e.y, e.x - e.y);
}
vec2 cExp(vec2 c)
{
  return exp(c.x) * cCis(c.y);
}

vec2 cLog(vec2 c)
{
  return vec2( log( cAbs(c) ), cArg(c) );
}
struct CP1_coord {
    vec2 z1;
    vec2 z2;
};
struct CP1_mat {
    mat2 real;
    mat2 imag;
};

CP1_mat sub(CP1_mat a, CP1_mat b){
    return CP1_mat(a.real-b.real, a.imag-b.imag);
}

CP1_mat make_CP1_mat(vec2 p,vec2 q,vec2 r,vec2 s) {
    return CP1_mat(mat2(p[0],q[0],
                        r[0],s[0]),
                   mat2(p[1],q[1],
                        r[1],s[1]));
}
void get_CP(in CP1_mat M, out vec2 a, out vec2 b,out vec2 c,out vec2 d){
    a = vec2(M.real[0][0], M.imag[0][0]);
    b = vec2(M.real[0][1], M.imag[0][1]);
    c = vec2(M.real[1][0], M.imag[1][0]);
    d = vec2(M.real[1][1], M.imag[1][1]);
}

CP1_mat matrix_mult(CP1_mat A, CP1_mat B){
    vec2 a,b,c,d,e,f,g,h;
    get_CP(A,a,b,c,d);
    get_CP(B,e,f,g,h);
    
    return make_CP1_mat(
        cMul(a,e) + cMul(b,g), cMul(a,f) + cMul(b,h),
        cMul(c,e) + cMul(d,g), cMul(c,f) + cMul(d,h)
    );
}


CP1_mat matrix_mult(CP1_mat a, vec2 b){
    vec2 a_0_0,a_0_1, a_1_1, a_1_0;
    get_CP(a,a_0_0,a_0_1,a_1_0,a_1_1);

    return make_CP1_mat(cMul(a_0_0,b), cMul(a_1_0,b),
                        cMul(a_0_1,b), cMul(a_1_1,b));
}

void matrix_mult(CP1_mat A, vec2 b1, vec2 b2, out vec2 out1, out vec2 out2){
    vec2 a,b,c,d;
    get_CP(A,a,b,c,d);

    out1 = cMul(a,b1) + cMul(b,b2);
    out2 = cMul(c,b1) + cMul(d,b2);
}

CP1_coord matrix_mult(CP1_mat A, CP1_coord b){
    vec2 z1,z2;
    matrix_mult(A, b.z1, b.z2, z1, z2);
    return CP1_coord(z1,z2);
}

CP1_mat matrix2_inv(CP1_mat a){
    vec2 a_0_0,a_0_1, a_1_1, a_1_0;
    get_CP(a,a_0_0,a_0_1,a_1_0,a_1_1);

    vec2 det = cDiv(vec2(1.0,0),
                    cMul(a_0_0,a_1_1) - cMul(a_1_0,a_0_1));

    CP1_mat mat = make_CP1_mat( a_1_1, -a_1_0,
                               -a_0_1,  a_0_0);
    return matrix_mult(mat,det);
}



vec4 draw2by2(mat2 mat, vec2 point, vec4 color){
    float scale = 50.0;
    point.y = 2. * scale - point.y;

    if (point.x < 0. || point.y < 0.){
        return vec4(0.,0.,0.,0.);
    }
    float val = 0.0;
    if (point.x < scale && point.y < scale){
        val = mat[0][0];
    } else if (point.x < scale*2. && point.y < scale){
        val = mat[0][1];
    }else if (point.y < scale*2. && point.x < scale){
        val = mat[1][0];
    } else if (point.y < scale*2. && point.x < scale*2.){
        val = mat[1][1];
    }
    if (val < 0.){
        color = color *-float(mod(point.x,2.0)>1.)-float(mod(point.y,2.0)>1.);
    }
    if (val > 1. && (mod(point.x,2.0)>1.)){
        color.y = val-1.;
    }
    return val * color;
}


vec4 draw(CP1_mat mat, vec2 point){
    vec4 ret = draw2by2(mat.real,point,vec4(1.,0.,0.,1.));
    ret += draw2by2(mat.imag,point-vec2(40.,0.),vec4(0.,1.,0.,1.));
    return ret;
}
vec4 draw(CP1_coord pt, vec2 point){
    vec4 ret = draw2by2(mat2(vec2(pt.z1.x,0.), vec2(pt.z2.x,0.)),point,vec4(1.,0.,0.,1.));
    ret += draw2by2(mat2(vec2(pt.z1.y,0.), vec2(pt.z2.y,0.)),point-vec2(50.,0.),vec4(0.,1.,0.,1.));
    return ret;
}

vec2 angles_from_pixel_coords(in vec2 point, in float x_size){
    //map from pixel coords to (0, 2*pi) x (-pi/2, pi/2) rectangle"""
    float y_size = x_size/2.0;  //assume equirectangular format
    return vec2(point[0] * 2.0*PI/x_size, 
                point[1] * PI/(y_size-1.0) - 0.5*PI);
}


vec2 pixel_coords_from_angles(in vec2 point, in float x_size){
    //map from (0, 2*pi) x (-pi/2, pi/2) rectangle to pixel coords"""
    float y_size = x_size/2.0;  //assume equirectangular format
    return vec2(point[0] * float(x_size)/(2.0*PI), (point[1] + 0.5*PI)* float(y_size-1.0)/PI);
}


vec2 angles_from_sphere(in vec3 point){
    //map from sphere in R^3 to (0, 2*pi) x (-pi/2, pi/2) rectangle (i.e. perform equirectangular projection)"""
    float longitude = atan(point.y,point.x);
    if (longitude < 0.0){
        longitude = longitude + PI*2.;
    }
    float r = sqrt(point.x*point.x+point.y*point.y);
    float latitude = atan(point.z,r);
    return vec2(longitude, latitude);
}

vec3 sphere_from_angles(in vec2 point){
    //map from (0, 2*pi) x (-pi/2, pi/2) rectangle to sphere in R^3 (i.e. perform inverse of equirectangular projection)"""
    float horiz_radius = cos(point.y);
    
    return vec3(horiz_radius*cos(point.x), 
                horiz_radius*sin(point.x),
                sin(point.y));
}

CP1_coord CP1_from_sphere(in vec3 point) {
    //map from sphere in R^3 to CP^1"""
    if (point.z < 0.0){
        return CP1_coord(vec2(point.x,point.y), vec2(1.0-point.z,0));
    }     else{
        return CP1_coord(vec2(1.0+point.z,0.0), vec2(point.x,-point.y));
    }
}


vec3 sphere_from_CP1(in CP1_coord point){
    //map from CP^1 to sphere in R^3"""
    if (length(point.z2) > length(point.z1)){
        vec2 z = cDiv(point.z1,point.z2);
        vec2 tmp = vec2(z.x, z.y); // x is real, y imag
        float denom = 1.0 + z.x*z.x + z.y*z.y;
        return vec3(2.0*z.x/denom, 2.0*z.y/denom, (denom - 2.0)/denom);
    } else {
        vec2 z = cConj(cDiv(point.z2,point.z1));
        float denom = 1.0 + z.x*z.x + z.y*z.y;
        return vec3(2.0*z.x/denom, 2.0*z.y/denom, (2.0 - denom)/denom);
    }
}

vec2 clamp(in vec2 point, in float x_size){
    //clamp to size of input, including wrapping around in the x direction""" 
    float y_size = x_size/2.0;      // assume equirectangular format
    vec2 ret = vec2(mod(point.x,x_size), point.y);
    if (point.y < 0.0){
        ret.y = 0.0;
    } else if ( point.y > y_size - 1.0) {
        ret.y = y_size - 1.0;
    } 
    return ret;
}
vec3 sphere_from_pixel_coords(in vec2 point, in float x_size){
    //map from pixel coords to sphere in R^3"""
    return sphere_from_angles(angles_from_pixel_coords(point, x_size));
}



CP1_mat inf_zero_one_to_triple(CP1_coord p,CP1_coord q,CP1_coord r){
    //"""returns SL(2,C) matrix that sends the three points infinity, zero, one to given input points p,q,r"""
    //### infinity = [1,0], zero = [0,1], one = [1,1] in CP^1
    CP1_mat M = make_CP1_mat(p.z1,q.z1,
                             p.z2,q.z2);
    CP1_mat Minv = matrix2_inv(M);
    vec2 mu, lam;
    matrix_mult(Minv,r.z1,r.z2,mu,lam);
    return make_CP1_mat(cMul(mu,p.z1), cMul(lam,q.z1),
                        cMul(mu,p.z2), cMul(lam,q.z2));
}
CP1_mat two_triples_to_SL(CP1_coord a1,CP1_coord b1,CP1_coord c1,CP1_coord a2,CP1_coord b2,CP1_coord c2){
    //"""returns SL(2,C) matrix that sends the three CP^1 points a1,b1,c1 to a2,b2,c2"""
    return matrix_mult( inf_zero_one_to_triple(a2,b2,c2), matrix2_inv(inf_zero_one_to_triple(a1,b1,c1) ) ) ;
}


CP1_mat three_points_to_three_points_pixel_coords(vec2 p1, vec2 q1, vec2 r1, vec2 p2, vec2 q2, vec2 r2, float x_size){
    //  """returns SL(2,C) matrix that sends the three pixel coordinate points a1,b1,c1 to a2,b2,c2"""
    CP1_coord p1_ = CP1_from_sphere(sphere_from_pixel_coords(p1,x_size));
    CP1_coord q1_ = CP1_from_sphere(sphere_from_pixel_coords(q1,x_size));
    CP1_coord r1_ = CP1_from_sphere(sphere_from_pixel_coords(r1,x_size));
    CP1_coord p2_ = CP1_from_sphere(sphere_from_pixel_coords(p2,x_size));
    CP1_coord q2_ = CP1_from_sphere(sphere_from_pixel_coords(q2,x_size));
    CP1_coord r2_ = CP1_from_sphere(sphere_from_pixel_coords(r2,x_size));

    return two_triples_to_SL(p1_,q1_,r1_,p2_,q2_,r2_);
}

vec3 get_vector_perp_to_p_and_q(vec3 p, vec3 q){
    //"""p and q are distinct points on sphere, return a unit vector perpendicular to both"""
    if (abs(dot(p,q) + 1.0) < 0.0001){ //### deal with the awkward special case when p and q are antipodal on the sphere
        if (abs(dot(p, vec3(1.0,0.0,0.0))) > 0.9999){ //#p is parallel to (1,0,0)
            return vec3(0.0,1.0,0.0);
        } else {
            return normalize(cross(p, vec3(1.0,0.0,0.0)));
        }
    } else {
        return normalize(cross(p, q));
    }
}

CP1_mat rotate_sphere_points_p_to_q(vec3 p, vec3 q){
    //"""p and q are points on the sphere, return SL(2,C) matrix rotating image of p to image of q on CP^1"""

    CP1_coord CP1p = CP1_from_sphere(p);
    CP1_coord CP1q = CP1_from_sphere(q);

    if (abs(dot(p,q) - 1.0) < 0.0001){
        return make_CP1_mat(vec2(1.0,0.0),vec2(0.0,0.0),
                            vec2(0.0,0.0),vec2(1.0,0.0));
    } else {
        vec3 r = get_vector_perp_to_p_and_q(p, q);
        CP1_coord CP1r = CP1_from_sphere(r);
        CP1_coord CP1mr =  CP1_from_sphere(-r);
        return two_triples_to_SL(CP1p, CP1r, CP1mr, CP1q, CP1r, CP1mr) ;
    }
}

CP1_mat rotate_pixel_coords_p_to_q(vec2 p_, vec2 q_, float x_size){
    //"""p and q are pixel coordinate points, return SL(2,C) matrix rotating image of p to image of q on CP^1"""
    vec3 p = sphere_from_pixel_coords(p_, x_size);
    vec3 q = sphere_from_pixel_coords(q_, x_size);
    return rotate_sphere_points_p_to_q(p,q);
}

CP1_mat rotate_around_axis_sphere_points_p_q(vec3 p,vec3 q, float theta){
    //"""p and q are points on sphere, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
    CP1_coord CP1p = CP1_from_sphere(p);
    CP1_coord CP1q = CP1_from_sphere(q);
    //assert dot(p,q) < 0.9999, "axis points should not be in the same place!"
    vec3 r = get_vector_perp_to_p_and_q(p, q);
    CP1_coord CP1r = CP1_from_sphere(r);
    CP1_mat M_standardise = two_triples_to_SL(CP1p, CP1q, CP1r,
                                              CP1_coord(vec2(0.0,0.0),vec2(1.0,0.0)),
                                              CP1_coord(vec2(1.0,0.0),vec2(0.0,0.0)),
                                              CP1_coord(vec2(1.0,0.0),vec2(1.0,0.0)));
    CP1_mat M_theta = make_CP1_mat(vec2(cos(theta),sin(theta)),vec2(0.0,0.0),
                                   vec2(0.0,0.0),vec2(1.0,0.0));// #rotate on axis through 0, infty by theta
    return matrix_mult( matrix_mult(matrix2_inv(M_standardise), M_theta), M_standardise );
}

CP1_mat rotate_around_axis_pixel_coords_p_q(vec2 p_,vec2 q_, float theta, float x_size){
    //"""p and q are pixel coordinate points, return SL(2,C) matrix rotating by angle theta around the axis from p to q"""
    

    vec3 p = sphere_from_pixel_coords(p_, x_size);
    vec3 q = sphere_from_pixel_coords(q_, x_size);
    return rotate_around_axis_sphere_points_p_q(p,q,theta);
}
CP1_mat rotate_around_axis_pixel_coord_p(vec2 p_,float theta, float x_size){
    //"""p is a pixel coordinate point, return SL(2,C) matrix rotating by angle theta around the axis from p to its antipode"""
    vec3 p = sphere_from_pixel_coords(p_, x_size);
    vec3 minus_p = -p;
    return rotate_around_axis_sphere_points_p_q(p,minus_p,theta);
}

CP1_mat zoom_in_on_pixel_coords(vec2 p, float zoom_factor, float x_size){
    //"""p is pixel coordinate point, return SL(2,C) matrix zooming in on p by a factor of scale"""
    //# Note that the zoom factor is only accurate at the point p itself. As we move away from p, we zoom less and less.
    //# We zoom with the inverse zoom_factor towards/away from the antipodal point to p.
    CP1_mat M_rot = rotate_pixel_coords_p_to_q( p, vec2(0.0,0.0), x_size);
    CP1_mat M_scl = make_CP1_mat(vec2(zoom_factor,0.0),vec2(0.0,0.0),
                                 vec2(0.0,0.0),    vec2(1.0,0.0)); //### zoom in on zero in CP^1
    return matrix_mult( matrix_mult(matrix2_inv(M_rot), M_scl), M_rot );
}

CP1_mat zoom_along_axis_sphere_points_p_q(vec3 p, vec3 q, float zoom_factor){
    //"""p and q are points on sphere, return SL(2,C) matrix zooming along axis from p to q"""
    CP1_coord CP1p = CP1_from_sphere(p);
    CP1_coord CP1q = CP1_from_sphere(q);
    //assert dot(p,q) < 0.9999   #points should not be in the same place
    vec3 r = get_vector_perp_to_p_and_q(p, q);
    CP1_coord CP1r = CP1_from_sphere(r);
    CP1_mat M_standardise = two_triples_to_SL(CP1p, CP1q, CP1r,
                                              CP1_coord(vec2(0.0,0.0),vec2(1.0,0.0)),
                                              CP1_coord(vec2(1.0,0.0),vec2(0.0,0.0)),
                                              CP1_coord(vec2(1.0,0.0),vec2(1.0,0.0)));
    CP1_mat M_theta = make_CP1_mat(vec2(zoom_factor,0.0),vec2(0.0,0.0),
                                   vec2(0.0,0.0),    vec2(1.0,0.0)); 
    return matrix_mult( matrix_mult(matrix2_inv(M_standardise), M_theta), M_standardise );
}
CP1_mat zoom_along_axis_pixel_coords_p_q(vec2 p_, vec2 q_, float zoom_factor, float x_size){
    //"""p and q are pixel coordinate points, return SL(2,C) matrix zooming along axis from p to q by zoom_factor"""
    //# This function does the same thing as zoom_in_on_pixel_coords, but with the 
    //# two given points instead of a single point and its antipodal point
    vec3 p = sphere_from_pixel_coords(p_, x_size);
    vec3 q = sphere_from_pixel_coords(q_, x_size);
    return zoom_along_axis_sphere_points_p_q(p,q,zoom_factor);
}

CP1_mat translate_along_axis_pixel_coords(vec2 p, vec2 q, vec2 r1, vec2 r2, float x_size){
    //"""Return SL(2,C) matrix translating/rotating on the axis from p to q, taking r1 to r2"""
    return three_points_to_three_points_pixel_coords(p,q,r1,p,q,r2, x_size);
}

vec2 apply_SL2C_elt_to_pt(CP1_mat M, vec2 pt_){
    CP1_mat Minv = matrix2_inv(M);
    vec2 pt = angles_from_pixel_coords(pt_, WIDTH);
    vec3 pt2 = sphere_from_angles(pt);
    CP1_coord pt3 = CP1_from_sphere(pt2);
    vec2 outa, outb;
    matrix_mult(Minv, pt3.z1,pt3.z2,outa,outb);
    CP1_coord pt4 = CP1_coord(outa,outb);
    vec3 pt5 = sphere_from_CP1(pt4);
    vec2 pt6 = angles_from_sphere(pt5);
    return pixel_coords_from_angles(pt6, WIDTH);
}
vec2 droste_effect(vec2 pt_,vec2 zoom_center_pixel_coords, float zoom_factor, float zoom_cutoff,
                   bool twist,float zoom_loop_value,float out_x_size){
    CP1_mat M_rot = rotate_pixel_coords_p_to_q(zoom_center_pixel_coords, vec2(0.,0.), out_x_size);
  CP1_mat M_rot_inv = matrix2_inv(M_rot);
    
    vec2 droste_factor = cDiv(cLog(complex(zoom_factor,0.)) + vec2(0., 2.*PI) , vec2(0., 2.*PI));
    
    vec2 pt = angles_from_pixel_coords(pt_, out_x_size);
    vec3 pt2 = sphere_from_angles(pt);
    CP1_coord pt3 = CP1_from_sphere(pt2);
    pt3 = matrix_mult(M_rot, pt3);
    
    pt = cDiv(pt3.z1,pt3.z2);
    pt = cLog(pt);
    if (twist){//:  # otherwise straight zoom
        pt = cMul(droste_factor,pt);
    }
    pt = complex(
        cReal(pt) + log(zoom_factor) * zoom_loop_value,
        cImag(pt)
    ); 
    pt = complex(
        mod(cReal(pt) + zoom_cutoff, log(zoom_factor)) - zoom_cutoff,
        cImag(pt)
    );
    pt = cExp(pt);
    pt3 = CP1_coord(
          pt,
          vec2(1.,0.)
        );// #back to projective coordinates
    pt3 = matrix_mult(M_rot_inv, pt3);
    pt2 = sphere_from_CP1(pt3);
    pt = angles_from_sphere(pt2);

    return pixel_coords_from_angles(pt, out_x_size);;
}
vec2 directionToPx(vec3 direction, float eye, float projection) {
    /*
    * Input: a direction.  +x = right, +y = up, +z = backward.
    *        an eye. left = 0, right = 1.
    *        a projection. see ProjectionEnum in JS file for enum
    * Output: a color from the video
    *
    * Bug alert: the control flow here may screw up texture filtering.
    */

    float theta = atan(direction.x, -1.0 * direction.z);
    float phi = atan(direction.y, length(direction.xz));

    /*
    * The Nexus 7 and the Moto X (and possibly many others) have
    * a buggy atan2 implementation that screws up when the numerator
    * (the first argument) is too close to zero.  (The 1e-4 is carefully
    * chosen: 1e-5 doesn't fix the problem.
    */
    if (abs(direction.x) < 1e-4 * abs(direction.z))
    theta = 0.5*PI * (1.0 - sign(-1.0 * direction.z));
    if (abs(direction.y) < 1e-4 * length(direction.xz))
    phi = 0.0;

    // Uncomment to debug the transformations.
    // return vec4(theta / (2. * PI) + 0.5, phi / (2. * PI) + 0.5, 0., 0.);

    if (projection == 0.) {
      // Projection == 0: equirectangular projection
      return vec2(mod(theta / (2.0*PI), 1.0), phi / PI + 0.5);
    } else {
      // Projection == 1: equirectangular top/bottom 3D projection
      eye = 1. - eye;
      return vec2(mod(theta / (2.0*PI), 1.0), ((phi / PI + 0.5) + eye)/ 2.);
    }
}

vec4 pxToColor(vec2 px){
    return texture2D(uSampler, px);
}
vec2 demo1(vec2 fragCoord){
    vec2 center = vec2(WIDTH/2.,HEIGHT/2.);
    CP1_mat transform = zoom_in_on_pixel_coords(center,4.,WIDTH);
    vec2 outpt = apply_SL2C_elt_to_pt(transform,fragCoord.xy);
    return outpt;
}

vec2 demo2(vec2 fragCoord){
    vec2 center = vec2(0.,0.);
    CP1_mat transform = zoom_in_on_pixel_coords(center,3.,1.);
    vec2 outpt = apply_SL2C_elt_to_pt(transform,fragCoord.xy);
    return outpt;
}
vec2 demo3(vec2 fragCoord){
    vec2 center1 = vec2(iMouse.xy);
    vec2 center2 = vec2(iMouse.x + 100.,iMouse.y);
    /*if (distance(fragCoord,center1) < 5. || distance(fragCoord,center2) < 5.){
        return vec4(1.0,1.0,0.0,1.0);
    }*/
    CP1_mat transform = rotate_around_axis_pixel_coords_p_q(center1,center2,iGlobalTime/2.0,WIDTH);
  vec2 outpt = apply_SL2C_elt_to_pt(transform,fragCoord.xy);
  return outpt;
}

vec2 demo4(vec2 fragCoord){
    vec2 center = iMouse.xy;//vec2(220.,150.);
    /*if (distance(fragCoord,center) < 5.){
        return vec2(1.0,1.0,0.0,1.0);
    }*/
    CP1_mat transform = rotate_pixel_coords_p_to_q(vec2(WIDTH/2.,HEIGHT/2.), center,WIDTH);
    vec2 outpt = apply_SL2C_elt_to_pt(transform,fragCoord.xy);
    return outpt;
}
vec2 demo5(vec2 fragCoord){
    vec2 center = vec2(0.,0.);//vec2(iMouse.x,HEIGHT/2.);
    return droste_effect(
        fragCoord,center, 7.*iMouse.y/HEIGHT + 1.3,1., false, 1.0, WIDTH);

}
vec4 testpattern(vec2 fragCoord){
  vec2 sphere = angles_from_sphere(sphere_from_angles(angles_from_pixel_coords(
      pixel_coords_from_angles(
          angles_from_sphere(
              sphere_from_CP1(
                  CP1_from_sphere(
                      sphere_from_pixel_coords(fragCoord.xy,WIDTH)))),WIDTH),WIDTH)));
  return vec4(sphere.x/(2.*PI),(sphere.y+PI/2.)/pi,0.0,1.0);
}
void main(void) {
    gl_FragColor = pxToColor(
                        demo2(
                            directionToPx(vDirection, eye, projection)*vec2(WIDTH,HEIGHT)
                            )/vec2(WIDTH,HEIGHT)
                        );
}
