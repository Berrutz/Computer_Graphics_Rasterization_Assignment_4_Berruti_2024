/*
Note: For the purpose of exercises 1 to 2, the world space, camera space, and
canonical viewing volume will be one and the same.

Ex.1: 
Load and Render a 3D model
The provided code rasterizes a triangle using the software rasterizer we studied in
the class. 
Extend the provided code to load the same scenes used in Assignment
3, and render them using rasterization in a uniform color. 

At this stage, you should see a correct silhouette of the object rendered (but no shading).

*/


// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cassert>

// Eigen for matrix operations example cross product
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Utilities for the Assignment
#include "ex1_raster.h"
#include <cstdint>

// JSON parser library (https://github.com/nlohmann/json)
#include "json.hpp"
using json = nlohmann::json;

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;


void load_scene(const std::string &filename, Program &program,  UniformAttributes &uniform, FrameBuffer &frameBuffer);
void load_off(const std::string &filename, vector<VertexAttributes> &vertex_attributes);
Matrix4f ortographic_matrix(Vector3f top_right_corner,float near , float far,float aspectRatio);
Matrix4f perspective_matrix(float near , float focal_lenght, float theta,float aspectRatio);
Matrix4f create_translation_matrix(const Vector3f& translation);
Matrix4f create_rotation_matrix(float angle, const Vector3f& axis);
Matrix4f create_scaling_matrix(const Vector3f& scale_factors);
Matrix4f create_model_matrix(const Vector3f& translation, 
float angle_x, float angle_y , float angle_z, const Vector3f& scale_factors);
Matrix4f create_rotation_matrix_x(float angle);
Matrix4f create_rotation_matrix_y(float angle);
Matrix4f create_rotation_matrix_z(float angle);
Matrix4f create_camera_matrix(Vector3f& eye,  Vector3f& target,  Vector3f& up);
void check_correctness_RGBA(Vector4f& vector);

int main(int argc, char *argv[]) 
{

	int height=1000;
	int width=1000;

	// The Framebuffer storing the image rendered by the rasterizer
	Eigen::Matrix<FrameBufferAttributes,Eigen::Dynamic,Eigen::Dynamic> 
	frameBuffer(width,height);  

	// Global Constants (empty in this example)
	UniformAttributes uniform;

	// Basic rasterization program
	Program program;

	// The vertex shader is the identity
	program.VertexShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
	{
		VertexAttributes out;

		Matrix4f Mfinal = uniform.projection_matrix * uniform.camera_matrix * uniform.model_matrix;
        out.position = Mfinal * va.position;
		out.position = out.position / out.position[3];  // to Cartesian 

        return out;
	};

	// The fragment shader uses a fixed color
	program.FragmentShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
	{
		FragmentAttributes out(uniform.color(0),uniform.color(1),uniform.color(2),uniform.color(3));
		return out;
	};

	// The blending shader converts colors between 0 and 1 to uint8
	program.BlendingShader = [](const FragmentAttributes& fa, const FrameBufferAttributes& previous)
	{
		return FrameBufferAttributes(fa.color[0]*255,fa.color[1]*255,fa.color[2]*255,fa.color[3]*255);
	};



	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " bunny.json" << std::endl;
		return 1;
	}

	load_scene(argv[1],program,uniform,frameBuffer);


	char const * img_ex1_folder = "../img/ex1/Base_case.png";
	vector<uint8_t> image;
	framebuffer_to_uint8(frameBuffer,image);
	stbi_write_png(img_ex1_folder, frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows()*4);
	
	return 0;
}

void check_correctness_RGBA(Vector4f& vec){
	if (( vec(0) < 0 || vec(0) > 1 ) ||
	    ( vec(1) < 0 || vec(1) > 1 ) ||
		( vec(2) < 0 || vec(2) > 1 ) ||
		( vec(3) < 0 || vec(3) > 1 )
	) 
	{
        throw std::runtime_error("Error in RGBA values");
    }
}

void load_scene(const std::string &filename,Program &program,UniformAttributes &uniform, FrameBuffer &frameBuffer){

	// Load json data from scene file
	std::ifstream in(filename);
	if (!in.is_open())
    {
        std::cerr << "Unable to open JSON file: " << filename << std::endl;
        return;
    }

	json data;
	in >> data;

	// Helper function to read a Vector3f from a json array
	auto read_vec3 = [] (const json &x) {
		return Vector3f(x[0], x[1], x[2]);
	};

	// Helper function to read a Vector3f from a json array
	auto read_vec4 = [] (const json &x) {
		return Vector4f(x[0], x[1], x[2],x[3]);
	};

	// Read the scene info
	auto scene = data["Scene"];
	uniform.ambient_light = read_vec3( scene["Ambient"]);

	// Read the camera info
	auto camera = data["Camera"];
	uniform.camera.is_perspective = camera["IsPerspective"] ;
	uniform.camera.position = read_vec3( camera["Position"] );
	uniform.camera.field_of_view = camera["FieldOfView"] ;
	Vector3f UpVector = read_vec3( camera["UpVector"] );
	Vector3f Target = read_vec3( camera["Target"] );			// Target Point that the camera is observing 
	uniform.camera.far = camera["Far"];
	uniform.camera.near = camera["Near"];
	uniform.camera.focal_lenght = camera["FocalLength"];

	// Creation of the Camera Matrix Mcam (	depends only on the viewer (camera extrinsic parameters) ) 
	uniform.camera_matrix = create_camera_matrix( uniform.camera.position , Target , UpVector );

	// Calculate the Ortographic or Perspective Matrix

	float h = frameBuffer.cols();
	float w = frameBuffer.rows();   
	float aspectRatio = float(h) / float(w);  	 	 

	if (uniform.camera.is_perspective){
		uniform.camera.far = uniform.camera.focal_lenght;
		uniform.projection_matrix = perspective_matrix( uniform.camera.near , uniform.camera.focal_lenght,
														uniform.camera.field_of_view, aspectRatio);
	}else{
		Vector3f top_right_corner = Vector3f(1, 1, uniform.camera.far);
		uniform.projection_matrix = ortographic_matrix(  top_right_corner , uniform.camera.near , uniform.camera.far , aspectRatio );
	}

	// Read all materials
	for (const auto &entry : data["Materials"]) {
		UniformAttributes::Material mat;
		mat.color_mat = read_vec4( entry["Color"] ) ;
		check_correctness_RGBA(mat.color_mat);
		uniform.materials.push_back(mat);
	}

	// Read all objects
	for (const auto &entry : data["Objects"]) {
		UniformAttributes::ObjectPtr object;
		
		if (entry["Type"] == "Mesh") {

			// Load mesh from a file
			std::string filename_off = std::string(DATA_DIR) + entry["Path"].get<std::string>();
			std::cout << filename_off << std::endl;

			MatrixXf vertices;                  
			MatrixXi facets;

			float angle_x = entry["angle_x"];			
			float angle_y = entry["angle_y"];
			float angle_z = entry["angle_z"];

			
			Vector3f translation = read_vec3(entry["translation"]);             
			Vector3f scale_factors = read_vec3(entry["scale"]);

			// Creation of the Modelling Matrix
			uniform.model_matrix = create_model_matrix(translation, 
			angle_x , angle_y , angle_z , scale_factors);

			// List of vertex of the Mesh
			vector<VertexAttributes> vertex_attributes = std::vector<VertexAttributes>();

			// read from the file the vertices and facets 
			load_off(filename_off,vertex_attributes); 

			// change color to rasterize based on the specific color of the Mesh 
			int material_value = entry["Material"];
			// check if the index exist in the list 
			if(material_value < 0 || material_value > uniform.materials.size() - 1 ){
				std::cout << " Material index to access (start from zero ) : " << material_value << std::endl;
				std::cout << " Material size  : " << uniform.materials.size() << std::endl;
				throw std::runtime_error("Error in Material Index value or lower than zero or bigger than Materials size");
			}
			uniform.color = uniform.materials[material_value].color_mat;

			// raster directly the object
			rasterize_triangles(program,uniform,vertex_attributes,frameBuffer);

		}else{
			// TODO
		}
	}

}


void load_off(const std::string &filename,vector<VertexAttributes> &v){

	MatrixXf Vertices;
	MatrixXi Facets;

	std::ifstream in(filename);
	std::string token;
	if (!in.is_open())
    {
        std::cerr << "Unable to open OFF file: " << filename << std::endl;
        return;
    }
	in >> token;
	int nv, nf, ne;
	in >> nv >> nf >> ne;


	Vertices.resize(nv, 3);
	Facets.resize(nf, 3);
	for (int i = 0; i < nv; ++i) {
		in >> Vertices(i, 0) >> Vertices(i, 1) >> Vertices(i, 2);
	}
	for (int i = 0; i < nf; ++i) {
		int s;
		in >> s >> Facets(i, 0) >> Facets(i, 1) >> Facets(i, 2);
		assert(s == 3);

		// create vertices and rasterize it for each facets 

		// i-th triangle
		Eigen::Vector3f a = Vertices.row(	Facets( i, 0)	);		// first vertice , 3 values
		Eigen::Vector3f b = Vertices.row(	Facets( i, 1)	);		// second vertice  
		Eigen::Vector3f c = Vertices.row(	Facets( i, 2)	);		// third vertice 
		v.push_back(VertexAttributes(a[0],a[1],a[2]));
		v.push_back(VertexAttributes(b[0],b[1],b[2]));
		v.push_back(VertexAttributes(c[0],c[1],c[2]));
	}
}

/////////////// Translation Matrix /////////////////////////////////////

Matrix4f create_translation_matrix(const Vector3f& translation)		// slide 8 page 17
{
    Matrix4f translation_matrix = Matrix4f::Identity();
    translation_matrix(0, 3) = translation.x();
    translation_matrix(1, 3) = translation.y();
    translation_matrix(2, 3) = translation.z();
    return translation_matrix;
}

/////////////// Scaling Matrix /////////////////////////////////////

Matrix4f create_scaling_matrix(const Vector3f& scale_factors)		// slide 8 page 17
{
    Matrix4f scaling_matrix = Matrix4f::Zero();
    scaling_matrix(0, 0) = scale_factors.x();
    scaling_matrix(1, 1) = scale_factors.y();
    scaling_matrix(2, 2) = scale_factors.z();
	scaling_matrix(3, 3) = 1;
    return scaling_matrix;
}

/////////////// Rotation Matrix /////////////////////////////////////

Matrix4f create_rotation_matrix(float angle_x,float angle_y,float angle_z) // slide 8 page 25
{
    Matrix4f rotation_matrix ;
    Matrix4f rotation_matrix_x = create_rotation_matrix_x(angle_x);
	Matrix4f rotation_matrix_y = create_rotation_matrix_y(angle_y);
	Matrix4f rotation_matrix_z = create_rotation_matrix_z(angle_z);

	rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;
	
    return rotation_matrix;
}


Matrix4f create_rotation_matrix_x(float angle){

	Matrix4f rotation_matrix = Matrix4f::Zero();

	// Compute sine and cosine of the angle
    float cos_angle = std::cos(angle);
    float sin_angle = std::sin(angle);

	rotation_matrix(0, 0) = 1;
    rotation_matrix(1, 1) = cos_angle;
    rotation_matrix(1, 2) = -sin_angle;
	rotation_matrix(2, 1) = sin_angle;
	rotation_matrix(2, 2) = cos_angle;
	rotation_matrix(3, 3) = 1;

	return rotation_matrix;
}

Matrix4f create_rotation_matrix_y(float angle){

	Matrix4f rotation_matrix = Matrix4f::Zero();
	
    float cos_angle = std::cos(angle);
    float sin_angle = std::sin(angle);

	rotation_matrix(0, 0) = cos_angle;
    rotation_matrix(0, 2) = sin_angle;
    rotation_matrix(1, 1) = 1;
	rotation_matrix(2, 0) = - sin_angle;
	rotation_matrix(2, 2) = cos_angle;
	rotation_matrix(3, 3) = 1;

	return rotation_matrix;
}

Matrix4f create_rotation_matrix_z(float angle){

	Matrix4f rotation_matrix = Matrix4f::Zero();
	
    float cos_angle = std::cos(angle);
    float sin_angle = std::sin(angle);

	rotation_matrix(0, 0) = cos_angle;
    rotation_matrix(0, 1) = - sin_angle;
    rotation_matrix(1, 0) = sin_angle;
	rotation_matrix(1, 1) = cos_angle;

	rotation_matrix(2, 2) = 1;
	rotation_matrix(3, 3) = 1;

	return rotation_matrix;
}

/////////////// MODELLING MATRIX //////////////////////////////////////////

Matrix4f create_model_matrix(const Vector3f& translation, 
float angle_x, float angle_y , float angle_z, const Vector3f& scale_factors)  // slide 8 page 17 - 3D transformation
{
    Matrix4f translation_matrix = create_translation_matrix(translation);
    Matrix4f rotation_matrix = create_rotation_matrix(angle_x,angle_y,angle_z);
    Matrix4f scaling_matrix = create_scaling_matrix(scale_factors);
	
    Matrix4f model_matrix = translation_matrix * rotation_matrix * scaling_matrix;
    return model_matrix;
}


/////////////////// CAMERA MATRIX //////////////////////////////////////////

Matrix4f create_camera_matrix(Vector3f& eye,  Vector3f& target,  Vector3f& up) {

	Vector3f u,v,w;
	Matrix3f R_transpose;
	Vector3f aux;

	w = (eye - target).normalized();  	  	  // z axis camera  
	u = w.cross(up).normalized();             // x axis camera
	v = u.cross(w);				  	      	  // y axis camera 

	std::cout << "w : " << w.transpose() << "| u : " << u.transpose() << "| v : " << v.transpose() << std::endl;

	Matrix3f R;
    R.col(0) << u.x(), u.y(), u.z();
	R.col(1) << v.x(), v.y(), v.z();
	R.col(2) << w.x(), w.y(), w.z();

	R_transpose = R.transpose();
	aux = R_transpose * ( - eye  ) ;

	// Construct the inverse camera matrix
    Matrix4f Mcam;
    Mcam << R_transpose(0, 0), R_transpose(0, 1), R_transpose(0, 2), aux.x(),
            R_transpose(1, 0), R_transpose(1, 1), R_transpose(1, 2), aux.y(),
            R_transpose(2, 0), R_transpose(2, 1), R_transpose(2, 2), aux.z(),
            0, 0, 0, 1;
	
	Vector4f origin_camera_system = ( Mcam * Vector4f( eye(0), eye(1), eye(2) , 1));
	Eigen::Vector4f expected(0, 0, 0, 1);
    assert(origin_camera_system.isApprox(expected) && "The origin is not well set (0, 0, 0, 1)");
    
	// return from World to camera 
    return Mcam;
}
   

///////////////////  ORTOGRAPHIC MATRIX /////////////////////////////////////// 

/// @brief This function will convert from the Camera Space to the Canonical view Volume taking care of the aspect ratio of the finale image
/// @param top_right_corner 
/// @param  
/// @param aspectRatio 
/// @return 
Matrix4f ortographic_matrix(Vector3f top_right_corner,float near , float far,float aspectRatio){

	
	float l,b,t,r,n,f ;

	f = far;											// z-position Far Plane  
	n = near;											// z-position Near Plane 
	t = top_right_corner[1] ;							
	r = top_right_corner[0] ;          				
	b = - t;
	l = - r;

	std::cout << "ORTO : r = " << r << " | t = " << t << " | f = " << f
	<< " | b = " << b << " | l = " << l << " | n = " << n << std::endl;

	// slide 12 page 2
	assert(n < 0 && f < 0 && "Near and Far Plane need to be negative because expressed in camera coordinate system that is watching in the negative direction");
	assert(n > f && " Near Plane must be lower then Far Plane ");
	assert(r >= l && " Right value must be higher than Left ");
	assert(t >= b && " Top value must be higher than Below ");

	Matrix4f ortographic = Matrix4f::Zero();
	ortographic(0, 0) = 2 / (r - l);
	ortographic(1, 1) = 2 / (t - b);
	ortographic(2, 2) = 2 / (n - f);
	ortographic(0, 3) = - (r + l) / (r - l);
	ortographic(1, 3) = - (t + b) / (t - b);
	ortographic(2, 3) = -(n + f) / (n - f); 
	ortographic(3, 3) = 1;

	// Manage  the aspect Ratio
	if (aspectRatio < 1)
            ortographic(0,0) *= aspectRatio;
    else
            ortographic(1,1) *= (1/aspectRatio);

	return ortographic; 
}


///////////////////  PERSPECTIVE MATRIX /////////////////////////////////////// 

Matrix4f perspective_matrix(float near , float focal_lenght, float theta,float aspectRatio){

	// The parameters can thus be found by ﬁxing n, f and theta . 
	// You can then compute b, t, and consequently all the other parameters needed to construct the transformation

	float l,b,t,r,n,f ;
	
	f = (- focal_lenght);				               					
	n = near;     													
	t = std::tan(theta/ 2.0) * std::abs(f);   						// slide 12 page 18
	r = t ;
	b = - std::tan(theta / 2.0) * std::abs(n); 
	l = b ;

	std::cout << "PRO r = " << r << " | t = " << t << " | f = " << f << " | b = " << b << " | l = " << l << " | n = " << n << std::endl;

	// slide 12 page 2
	assert(n < 0 && f < 0 && "Near and Far Plane need to be negative because expressed in camera coordinate system that is watching in the negative direction");
	assert(n > f && " Near Plane must be lower then Far Plane ");
	assert(r >= l && " Right value must be higher than Left ");
	assert(t >= b && " Top value must be higher than Below ");

	Matrix4f ortographic = Matrix4f::Zero();
	ortographic(0, 0) = 2 / (r - l);
	ortographic(1, 1) = 2 / (t - b);
	ortographic(2, 2) = 2 / (n - f);
	ortographic(0, 3) = - (r + l) / (r - l);
	ortographic(1, 3) = - (t + b) / (t - b);
	ortographic(2, 3) = -(n + f) / (n - f); 
	ortographic(3, 3) = 1;

	// Slide 12 page 17
	Matrix4f P = Matrix4f::Zero();
	P(0, 0) = n;
	P(1, 1) = n;
	P(2, 2) = n+f;
	P(2, 3) = - f * n;
	P(3, 2) = 1;

	Matrix4f MProj = ortographic * P;

	// Manage  the aspect Ratio
	if (aspectRatio < 1)
            MProj(0,0) *= aspectRatio;
    else
            MProj(1,1) *= (1/aspectRatio);

	return MProj ; 
}
