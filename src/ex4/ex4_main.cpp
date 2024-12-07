/*

Ex.4: Camera

1.
Implement a perspective camera and add support for multiple resolutions: the
cameras should take into account the size of the framebuffer, properly adapting
the aspect ratio to not distort the image whenever the framebuffer is resized. To
check for correctness, we recommend to render a cube in wireframe mode.

2.
Note that for the shading to be correct, the lighting equation need to be computed
in the camera space (sometimes called the eye space). This is because in the
camera space the position of the camera center is known (it is (0,0,0)), but in the
canonical viewing volume normals can be distorted by the perspective projection,
so shading will be computed incorrectly.

*/

// gif
#include <gif.h>


// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

// Eigen for matrix operations example cross product
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Utilities for the Assignment
#include "ex4_raster.h"
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
void load_off(const std::string &filename, vector<VertexAttributes> &vertex_attributes_flat,
vector<VertexAttributes> &vertex_attributes_wire,
vector<VertexAttributes> &vertex_attributes_per_vertex,
UniformAttributes &uniform);
Matrix4f ortographic_matrix(Vector3f top_right_corner,float near , float far,float aspectRatio);
Matrix4f perspective_matrix(float near , float focal_lenght,float theta,float aspectRatio);

Matrix4f create_translation_matrix(const Vector3f& translation);
Matrix4f create_rotation_matrix(float angle, const Vector3f& axis);
Matrix4f create_scaling_matrix(const Vector3f& scale_factors);
Matrix4f create_model_matrix(const Vector3f& translation, 
float angle_x, float angle_y , float angle_z, const Vector3f& scale_factors);
Matrix4f create_rotation_matrix_x(float angle);
Matrix4f create_rotation_matrix_y(float angle);
Matrix4f create_rotation_matrix_z(float angle);

Matrix4f create_camera_matrix(Vector3f& eye,  Vector3f& target,  Vector3f& up);

Vector3f compute_center_3d_model(const MatrixXf& vertices, const MatrixXi& facets);
void check_correctness_RGBA(Vector4f& vector);

//// Normal /////
void compute_face_normals(const MatrixXf &vertices, const MatrixXi &facets, MatrixXf &face_normals,Vector3f center_of_the_model);
void compute_per_vertex_normals(const MatrixXf &vertices, const MatrixXi &facets, const MatrixXf &face_normals, MatrixXf &vertex_normals);
enum ShadingMode { WIREFRAME, FLAT_SHADING, PER_VERTEX_SHADING };
ShadingMode shadingMode = WIREFRAME; 										// Default shading mode


///////////////////////// GIF Params ////////////////////////////////////

// Animation Parameters
const int num_frames = 25; 					// Number of frames
const float animation_speed = 0.1; 			// Animation speed

// Name of the gif
GifWriter g;

////////////////////////////////////////////////////////////////////



int main(int argc, char *argv[]) 
{

	int height=750;
	int width=1000;

	if (argc < 2 || argc > 3) {
		std::cerr << "Usage: " << argv[0] << " bunny.json" << " wire|flat|vertex " << std::endl;
		return 1;
	} else if (argc == 2) {
		shadingMode = WIREFRAME;
		std::cout << "Mode set to (default): " << shadingMode << std::endl;
		std::cout << "File chosen is: " << argv[1] << std::endl;
	} else if (argc == 3) {
		std::string mode = argv[2];
		switch(mode[0]) {
			case 'w':
				shadingMode = WIREFRAME;
				break;
			case 'f':
				shadingMode = FLAT_SHADING;
				break;
			case 'v':
				shadingMode = PER_VERTEX_SHADING;
				break;
			default:
				std::cerr << "Invalid shading mode. Available options: wire, flat, vertex." << std::endl;
				return 1;
		}
		std::cout << "Mode set to: " << shadingMode << std::endl;
		std::cout << "File chosen is: " << argv[1] << std::endl;
	}

	/****   GIF PART  *********/

	int delay = 25; // Delay in milliseconds between frames
	std::string mode = (shadingMode == WIREFRAME) ? "Wireframe" :
    (shadingMode == FLAT_SHADING) ? "Flat" : "Vertex";
	std::string basePath = "../img/ex4/Animation_";
	std::string fullPath = basePath + mode + ".png";

	const char* fileName = fullPath.c_str();

	Eigen::Matrix<FrameBufferAttributes,Eigen::Dynamic,Eigen::Dynamic> 
	frameBuffer(width,height);  
	GifBegin(&g, fileName,frameBuffer.rows(), frameBuffer.cols(), delay);
	
	float rotation_on_x_frame = 0;

	/*************************/

	for (int frame = 0; frame < num_frames; ++frame) {
		std::cout << "frame : " << frame << std::endl;

		// Global Constants (empty in this example)
		UniformAttributes uniform;

		// Basic rasterization program
		Program program;

		// The vertex shader is the identity
		program.VertexShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
		{
			VertexAttributes out;

			Vector4f ModelTransformedPosition;
			Vector4f CameraTransformedPosition;
			Vector4f ProjectionTransformedPosition;

			// ROTATE ROTATION MATRIX BASED ON FRAMES AND ROTATION SPEED AROUND BARICENTER
			Vector3f center = uniform.center_of_the_model;

			Matrix4f translate_to_center = create_translation_matrix(-center);
			Matrix4f translate_back_to_point = create_translation_matrix(center);
			double theta = uniform.frame * uniform.speed;
			Matrix4f rotation_on_z_frame = create_rotation_matrix_z(theta);

			float translation_speed = -0.05;
			Vector3f trans = Vector3f(0,0, uniform.frame * translation_speed );
			Matrix4f translation_towards_camera = create_translation_matrix(trans);

			// MODELLING TRASFORMATION
			Matrix4f model = uniform.model_matrix;
			Matrix4f Rotate_around_baricenter = translate_back_to_point * rotation_on_z_frame * translate_to_center;
			ModelTransformedPosition = translation_towards_camera * model * Rotate_around_baricenter * va.position;

			// CAMERA TRASFORMATION
			Matrix4f camera = uniform.camera_matrix;
			CameraTransformedPosition = camera * ModelTransformedPosition;

			// PROJECTION TRASFORMATION
			Matrix4f projection = uniform.projection_matrix;
			ProjectionTransformedPosition = projection * CameraTransformedPosition;

			/* CHECK THAT THE INVERSE WORK */
			Vector4f aux  = uniform.projection_matrix_inverse * ProjectionTransformedPosition;
			float tolerance = 1e-6;
			assert( (aux - CameraTransformedPosition).norm() < tolerance && "ERROR in Inverse");

			Vector4f FinalNormal;
			if(shadingMode == PER_VERTEX_SHADING){
					FinalNormal = va.vertex_normal;   					
					out.vertex_normal  = FinalNormal;
			}else{
					FinalNormal = va.face_normal; 
					out.face_normal  = FinalNormal;
			} 

			Vector4f FinalPosition = ProjectionTransformedPosition;  // Transformed in Projection System
			out.position = FinalPosition;

			return out;
		};

		// The fragment shader uses a fixed color
		program.FragmentShader = [](const VertexAttributes& va, const UniformAttributes& uniform)
		{
			////////////// ILLUMINATION EQUATION ////////////////////////////
			
			UniformAttributes::Material mat = uniform.materials[uniform.object_material_index];  
			Vector4f C;
			

			Vector3f camera_in_camera_space = Vector3f(0,0,0); 
			Vector4f point_normal ;
			
			if(shadingMode == FLAT_SHADING  || shadingMode == PER_VERTEX_SHADING ){

				Vector4f point_in_camera_space_4d = uniform.projection_matrix_inverse * va.position ;
				Vector3f point_in_camera_space = Vector3f(point_in_camera_space_4d(0),point_in_camera_space_4d(1),point_in_camera_space_4d(2));

				if(shadingMode == PER_VERTEX_SHADING){
					point_normal = Vector4f(va.vertex_normal(0),va.vertex_normal(1),va.vertex_normal(2),0);
				}else{
					point_normal = Vector4f(va.face_normal(0),va.face_normal(1),va.face_normal(2),0);
				}


				Vector4f ambient_color_4d = uniform.ambient_light.array() * mat.ambient_color.array(); //Vector4f ambient_color = uniform.ambient_light.cwiseProduct(mat.ambient_color);
				Vector3f ambient_color = Vector3f(ambient_color_4d(0),ambient_color_4d(1),ambient_color_4d(2));


				Vector3f lights_color = Vector3f(0, 0, 0);

				// Contribution of each Light
				for (const UniformAttributes::Light &light : uniform.lights) {
						
						Vector4f light_in_camera_space_4d = uniform.camera_matrix * light.Liposition;
						Vector3f light_in_camera_space = Vector3f(light_in_camera_space_4d(0),light_in_camera_space_4d(1),light_in_camera_space_4d(2));

						assert(light_in_camera_space_4d[3] == 1 && "ERROR - Point light_in_camera_space not properly normalized ");

						Vector4f normal_in_camera_space_4d = point_normal;
					
						normal_in_camera_space_4d[3] = 0;
						
						Vector3f normal_in_camera_space = Vector3f(normal_in_camera_space_4d(0),normal_in_camera_space_4d(1),normal_in_camera_space_4d(2));
					
						Vector3f Li = (light_in_camera_space - point_in_camera_space).normalized();

						Vector3f N = normal_in_camera_space ;

						// Diffuse contribution
						double cos_theta_Li_N = Li.dot(N);
						
						Vector3f diffuse_color = Vector3f(mat.diffuse_color(0),mat.diffuse_color(1),mat.diffuse_color(2));
						Vector3f diffuse = diffuse_color * std::max( cos_theta_Li_N , 0.0);

						// TODO (Assignment 2, specular contribution)
						Vector3f v = (camera_in_camera_space - point_in_camera_space).normalized();

						Vector3f h = ( v + Li ).normalized();

						float cos_theta_n_h = N.dot(h);
						float p = mat.specular_exponent;
						Vector3f specular_color = Vector3f(mat.specular_color(0),mat.specular_color(1),mat.specular_color(2));
						Vector3f specular =  specular_color * (std::pow( std::max(cos_theta_n_h , 0.0f) , p)); 

						Vector3f light_intensity = Vector3f(light.intensity(0),light.intensity(1),light.intensity(2));
						Vector3f D = light_in_camera_space - point_in_camera_space;
						lights_color += (diffuse + specular).cwiseProduct(light_intensity) /  D.squaredNorm();
					}

				// Rendering equation
				Vector3f C_3d = (ambient_color + lights_color);
				C = Vector4f(C_3d(0),C_3d(1),C_3d(2),1.0);
				
			}else{

				C = mat.color_mat;
				point_normal = Vector4f(0,0,0,0); 
			}

			if(C(0) + C(1) + C(2) > 1){
				float sum = C(0) + C(1) + C(2);
				C = C / sum;
				C(3) = 1;
			}

			FragmentAttributes out(C(0),C(1),C(2),C(3));   

			Vector4f FinalPosition = va.position;                           // already in projection space 
			
			out.position = FinalPosition;

			return out;
		};

		// The blending shader converts colors between 0 and 1 to uint8
		program.BlendingShader = [](const FragmentAttributes& fa, const FrameBufferAttributes& previous)
	{
		
		if (fa.position[2] < previous.depth)
		{
			FrameBufferAttributes out(fa.color[0]*255, fa.color[1]*255, fa.color[2]*255, fa.color[3]*255);
			out.depth = fa.position[2];
			return out;
		}
		return previous;
		
	};


		Eigen::Matrix<FrameBufferAttributes,Eigen::Dynamic,Eigen::Dynamic> 
		frameBuffer(width,height);  

		uniform.frame = frame;
		uniform.speed = animation_speed;
		rotation_on_x_frame = frame * animation_speed;
		uniform.rotation_on_x_frame = rotation_on_x_frame;

		load_scene(argv[1],program,uniform,frameBuffer);

		vector<uint8_t> image;
		framebuffer_to_uint8(frameBuffer,image);

		// write to gif
		GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
	}

	GifEnd(&g);

	return 0;
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

	auto read_vec4 = [] (const json &x) {
		return Vector4f(x[0], x[1], x[2],1.0f);
	};

	// Read the scene info
	auto scene = data["Scene"];
	uniform.ambient_light = read_vec4( scene["Ambient"]);


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
	uniform.camera_matrix_inv_transpose = uniform.camera_matrix.inverse().transpose();  // ok 

	std::cout << "Camera Matrix " << std::endl;
	std::cout << uniform.camera_matrix << std::endl;


	float h = frameBuffer.cols();
	float w = frameBuffer.rows();   
	float aspectRatio = float(h) / float(w);  	 	 

	std::cout << " w = " << w << 
	" | h = " << h << " | aspectRatio = " << 1/aspectRatio << std::endl; 
	

	// Compute the Projection Matrix ( Ortographic or Perspective )
	if (uniform.camera.is_perspective){

		uniform.camera.far = uniform.camera.focal_lenght;
		uniform.projection_matrix = perspective_matrix( uniform.camera.near , uniform.camera.focal_lenght,
														uniform.camera.field_of_view, aspectRatio);
		uniform.projection_matrix_inverse = uniform.projection_matrix.inverse();
		uniform.projection_matrix_inv_transpose = uniform.projection_matrix_inverse.transpose();
	
	}else{

		uniform.top_right_corner = Vector3f(1, 1, uniform.far);
		uniform.projection_matrix = ortographic_matrix(  uniform.top_right_corner , 
		uniform.camera.near , uniform.camera.far , aspectRatio );
		uniform.projection_matrix_inverse = uniform.projection_matrix.inverse();
		uniform.projection_matrix_inv_transpose = uniform.projection_matrix_inverse.transpose();

		std::cout << "PROJECTION MATRIX ORTOGRAPHIC: \n" << uniform.projection_matrix << std::endl;
		
	}

	// Read all materials
	for (const auto &entry : data["Materials"]) {
		UniformAttributes::Material mat;
		mat.color_mat = read_vec4(entry["Color"]);  
		mat.ambient_color = read_vec4(entry["Ambient"]);

		mat.diffuse_color = read_vec4(entry["Diffuse"]);
		mat.specular_color = read_vec4(entry["Specular"]);
		mat.specular_exponent = entry["Shininess"];

		check_correctness_RGBA(mat.ambient_color);
		check_correctness_RGBA(mat.color_mat);

		uniform.materials.push_back(mat);
	}

	// Read all lights
	for (const auto &entry : data["Lights"]) {
		UniformAttributes::Light light;
		light.Liposition << read_vec4(entry["Position"]);  // Position of light in World Coordinate System 
		light.intensity = read_vec4(entry["Color"]);
		uniform.lights.push_back(light);
	}

	// Read all objects
	for (const auto &entry : data["Objects"]) {
		UniformAttributes::ObjectPtr object;

		if (entry["Type"] == "Mesh") {

			// Retrive name of the .off file
			std::string filename_off = std::string(DATA_DIR) + entry["Path"].get<std::string>();
			std::cout << filename_off << std::endl;

			MatrixXf vertices;                  
			MatrixXi facets;

			// Read the rotation angle of the object in radians (theta)
 			float angle_x = entry["angle_x"];			
			float angle_y = entry["angle_y"];
			float angle_z = entry["angle_z"];

			// Read scale of the Mesh and translation
			Vector3f translation = read_vec3(entry["translation"]);             
			Vector3f scale_factors = read_vec3(entry["scale"]);

			// Creation of the Modelling Matrix
			uniform.model_matrix = create_model_matrix(translation, angle_x , angle_y , angle_z , scale_factors);
			uniform.model_matrix_inv_transpose = uniform.model_matrix.inverse().transpose();

			std::cout << "Model Matrix " << std::endl;
			std::cout << uniform.model_matrix << std::endl;
			

			// change color to rasterize based on the specific color of the Mesh 
			int material_value = entry["Material"];
			uniform.object_material_index = material_value;

			// List of vertex of the Mesh in each type of rendering 
			vector<VertexAttributes> vertex_attributes_flat = std::vector<VertexAttributes>();
			vector<VertexAttributes> vertex_attributes_wire = std::vector<VertexAttributes>();
			vector<VertexAttributes> vertex_attributes_per_vertex = std::vector<VertexAttributes>();

			load_off(filename_off, 
			vertex_attributes_flat,
			vertex_attributes_wire,
			vertex_attributes_per_vertex,
			uniform); 

			// WIREFRAME , 3 line for each triangle
			if (shadingMode == WIREFRAME ) {

				// Draw lines between vertices
				float thickness =  1.0;
				rasterize_lines(program, uniform, vertex_attributes_wire,thickness, frameBuffer);

			}else if(shadingMode == FLAT_SHADING){
			
				rasterize_triangles(program,uniform,vertex_attributes_flat,frameBuffer);

				shadingMode = WIREFRAME;   
				float thickness =  1; 
				uniform.object_material_index = 1;  // change color of the WIREFRAME
				rasterize_lines(program,uniform,vertex_attributes_wire,thickness,frameBuffer);
				
				shadingMode = FLAT_SHADING; // return to FLAT to be consistent for other frames
			
			}else{
				 
				float thickness =  1.0;
				rasterize_triangles(program,uniform,vertex_attributes_per_vertex,frameBuffer);
			}

		}else{
			// TODO
		}
	}


}


void load_off(const std::string &filename, 
vector<VertexAttributes> &vertex_attributes_flat,
vector<VertexAttributes> &vertex_attributes_wire,
vector<VertexAttributes> &vertex_attributes_per_vertex,
UniformAttributes &uniform){

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

	}

	// Face and Per Vertex Normal
	MatrixXf Face_normals;
	MatrixXf Per_Vertex_Normal;

	Vector3f center_of_the_model = compute_center_3d_model(Vertices, Facets);

	uniform.center_of_the_model = center_of_the_model;
	
	// compute per face normal 
	compute_face_normals(Vertices,Facets,Face_normals,center_of_the_model);
	// computer per vertex normal  
	compute_per_vertex_normals(Vertices,Facets,Face_normals,Per_Vertex_Normal);

	for (int i = 0; i < nf; ++i) {
		// create vertices and rasterize it for each facets 

		// i-th triangle
		Eigen::Vector3f a = Vertices.row(	Facets( i, 0)	);		// first vertice , 3 values
		Eigen::Vector3f b = Vertices.row(	Facets( i, 1)	);		// second vertice  
		Eigen::Vector3f c = Vertices.row(	Facets( i, 2)	);		// third vertice  

		if (shadingMode == FLAT_SHADING) {

			Vector3f face_normal = Face_normals.row( i );
			Vector4f face_normal_4d;
    		face_normal_4d << face_normal, 0.0f;

			vertex_attributes_flat.push_back(VertexAttributes(a[0],a[1],a[2],1.0,face_normal_4d));
			vertex_attributes_flat.push_back(VertexAttributes(b[0],b[1],b[2],1.0,face_normal_4d));
			vertex_attributes_flat.push_back(VertexAttributes(c[0],c[1],c[2],1.0,face_normal_4d));

			// Wireframe also
			vertex_attributes_wire.push_back(VertexAttributes(a[0],a[1],a[2]));
			vertex_attributes_wire.push_back(VertexAttributes(b[0],b[1],b[2]));

			vertex_attributes_wire.push_back(VertexAttributes(b[0],b[1],b[2]));
			vertex_attributes_wire.push_back(VertexAttributes(c[0],c[1],c[2]));

			vertex_attributes_wire.push_back(VertexAttributes(c[0],c[1],c[2]));
			vertex_attributes_wire.push_back(VertexAttributes(a[0],a[1],a[2]));
			
        } 
		else if (shadingMode == PER_VERTEX_SHADING) {

			Vector4f ZeroVectorFace =  Vector4f(0,0,0,0);
			
			// Normal first vertice 
			int a_index = Facets( i , 0); // index of the 1^ vertice
			Vector3f normal_a = Per_Vertex_Normal.row( a_index );  
			Vector4f normal_vertex_4d_a;
    		normal_vertex_4d_a << normal_a, 0.0f;
			
			// Normal second vertice 
			int b_index = Facets( i , 1); // index of the 2^ vertice
			Vector3f normal_b = Per_Vertex_Normal.row( b_index );  
			Vector4f normal_vertex_4d_b;
    		normal_vertex_4d_b << normal_b, 0.0f;
		
			// Normal third vertice 
			int c_index = Facets( i , 2); // index of the 3^ vertice
			Vector3f normal_c = Per_Vertex_Normal.row( c_index );  
			Vector4f normal_vertex_4d_c;
    		normal_vertex_4d_c << normal_c, 0.0f;

			vertex_attributes_per_vertex.push_back(VertexAttributes(a[0],a[1],a[2],1.0,ZeroVectorFace,normal_vertex_4d_a));
			vertex_attributes_per_vertex.push_back(VertexAttributes(b[0],b[1],b[2],1.0,ZeroVectorFace,normal_vertex_4d_b));
			vertex_attributes_per_vertex.push_back(VertexAttributes(c[0],c[1],c[2],1.0,ZeroVectorFace,normal_vertex_4d_c));

        }else{
			// wIREFRAME
			vertex_attributes_wire.push_back(VertexAttributes(a[0],a[1],a[2]));
			vertex_attributes_wire.push_back(VertexAttributes(b[0],b[1],b[2]));

			vertex_attributes_wire.push_back(VertexAttributes(b[0],b[1],b[2]));
			vertex_attributes_wire.push_back(VertexAttributes(c[0],c[1],c[2]));

			vertex_attributes_wire.push_back(VertexAttributes(c[0],c[1],c[2]));
			vertex_attributes_wire.push_back(VertexAttributes(a[0],a[1],a[2]));
		}
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
float angle_x, float angle_y , float angle_z, const Vector3f& scale_factors)
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

	w = (eye - target).normalized();   // z axis camera
	u = w.cross(up).normalized();      // x axis camera
	v = u.cross(w);				       // y axis camera  

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
/// @param near
/// @param far   
/// @param aspectRatio 
/// @return 
Matrix4f ortographic_matrix(Vector3f top_right_corner,float near , float far, float aspectRatio){
	// Here we dont'have theta angle so we decide the top right corner of the Image Plane
	
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
	assert(n > f && " Near Plane must be lower then Far Plane in Camera coordinate system  ");
	assert(r >= l && " Right value must be higher than Left ");
	assert(t >= b && " Top value must be higher than Below ");

	Matrix4f ortographic = Matrix4f::Zero();
	ortographic(0, 0) = 2 / (r - l);
	ortographic(1, 1) = 2 / (t - b);
	ortographic(2, 2) = 2 / (n - f);
	ortographic(0, 3) = - (r + l) / (r - l);
	ortographic(1, 3) = - (t + b) / (t - b);
	ortographic(2, 3) = - (n + f) / (n - f); 
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
	ortographic(2, 3) = - (n + f) / (n - f); 
	ortographic(3, 3) = 1;

	// P slide 12 page 17
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



//////////////// NORMAL FACE AND PER-VERTEX NORMAL ///////////////////////

void compute_face_normals(const MatrixXf &vertices, const MatrixXi &facets, 
	MatrixXf &face_normals , Vector3f center_of_the_model) {

    face_normals.resize(facets.rows(), 3);
	// For each traingle (facets) made of 3 vertices 
	// save at the index related to the face the normal of the triangle
    
	for (int i = 0; i < facets.rows(); ++i) {
        Vector3f v0 = vertices.row(facets(i, 0));
        Vector3f v1 = vertices.row(facets(i, 1));
        Vector3f v2 = vertices.row(facets(i, 2));
        Vector3f normal = (v2 - v0).cross(v1 - v0).normalized();    
		

		////////////// computing correctly the normal  //////////////////////77
		
		Vector3f to_center = center_of_the_model - v0;

		// check correcteness
		float tolerance = 1e-6;
		assert( ((v0 + to_center - center_of_the_model).norm() < tolerance)  && "ERROR - v0 + vector_to_center = center_of_the_model " );
 
		if (normal.dot(to_center) > 0){   //the scalar product is simmetric
			normal = (v1 - v0).cross(v2 - v0).normalized();
		}

		assert( normal.dot(to_center) <= 0 && "Face Normal not properly set ");  // we assure that the normal point in different direction

        face_normals.row(i) = normal;
    }
}


void compute_per_vertex_normals(const MatrixXf &vertices, const MatrixXi &facets, const MatrixXf &face_normals, MatrixXf &vertex_normals) {
    vertex_normals.resize(vertices.rows(), 3);
    vertex_normals.setZero();
    vector<int> count(vertices.rows(), 0);

	// Now we already have the normal related to each Triangle (triplets of vertices)
	// Each vertice has it's own index 
	// So we iterate on the Traingle and for each index of the 3 vertex we  
	// first take the index of the vertice 
	// second we save the normal of the facet we are iterating on and sum the normal of the triangle
	// we do it for each vertex in the triangle
	// And in another structure we save how much triangle are related to the vertex 
	// So when we encounted another time the same vertice (it's index) in another triangle we sum the normal related to the other triangle 
	// and increase the number of triangle that this vertex touch 

    for (int i = 0; i < facets.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int idx = facets(i, j);
            vertex_normals.row(idx) += face_normals.row(i);
            count[idx]++;
        }
    }

	// Now each vertice has the sum of the normal related to all the facets that he touches .
	// So now we average the summed normal with the number of triangle related to the vertice
    for (int i = 0; i < vertices.rows(); ++i) {
        if (count[i] > 0) {
            vertex_normals.row(i) /= count[i];
			vertex_normals.row(i).normalize();
        }
    }
}


//////////////////////  Center of the Model ////////////////////////////////////////


Vector3f compute_center_3d_model(const MatrixXf& vertices, const MatrixXi& facets) {

    Vector3f center = Vector3f::Zero();
    int numFacets = facets.rows();

    for (int i = 0; i < numFacets; ++i) {
        
        Vector3f v0 = vertices.row(facets(i, 0));
        Vector3f v1 = vertices.row(facets(i, 1));
        Vector3f v2 = vertices.row(facets(i, 2));

        Vector3f faceCenter = (v0 + v1 + v2) / 3.0f;

        center += faceCenter;
    }

    center /= numFacets;

    return center;
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