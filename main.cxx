#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace sdk {
	using namespace cv;
	using namespace dnn;
	using namespace std;

	struct model_conf_t {
		std::string m_model_path;
		std::string m_names_path;
		std::int32_t m_blob_x;
		std::int32_t m_blob_y;
		std::float_t m_hit_min;
		std::float_t m_iou_max;
		std::string m_model_conf;
		std::string m_model_type;
	};

	struct model_t {
		model_t( ) = default;
		model_t(
			const model_conf_t conf,
			const std::int32_t backend = cv::dnn::Backend::DNN_BACKEND_OPENCV,
			const std::int32_t target = cv::dnn::Target::DNN_TARGET_CPU
		) {
			m_net = cv::dnn::readNet( conf.m_model_path, conf.m_model_conf, conf.m_model_type );
			if ( m_net.empty( ) ) {
				std::cerr << "[dnn] failed to initialise net\n";
				return;
			}

			m_net.setPreferableBackend( backend );
			m_net.setPreferableTarget( target );

			std::fstream names{ conf.m_names_path, std::ios::in };
			if ( !names.is_open( ) ) {
				std::cerr << "[dnn] failed to read names\n";
				return;
			}

			std::string cur_name{};
			while ( std::getline( names, cur_name ) )
				m_names.push_back( cur_name );

			m_config = conf;
		}

		std::int8_t detect_image(
			const std::string_view image_path,
			const std::int32_t image_flags = cv::ImreadModes::IMREAD_COLOR
		) {
			m_image = cv::imread( image_path.data( ), image_flags );
			if ( m_image.empty( ) ) {
				std::cerr << "[dnn] failed to read image\n";
				return 0;
			}

			std::int32_t blob_x{ m_config.m_blob_x };
			std::int32_t blob_y{ m_config.m_blob_y };

			m_blob = cv::dnn::blobFromImage( m_image, 1.f / 255.f, cv::Size{ blob_x, blob_y } );
			if ( m_blob.empty( ) ) {
				std::cerr << "[dnn] failed to create blob\n";
				return 0;
			}

			const auto out_names = [](
				const cv::dnn::Net& cv_net
			) {
				static std::vector< std::string >names{};
				if ( names.empty( ) ) {
					std::vector< std::int32_t >out_layers{ cv_net.getUnconnectedOutLayers( ) };
					std::vector< std::string >layer_names{ cv_net.getLayerNames( ) };

					names.resize( out_layers.size( ) );
					for ( std::size_t i{}; i < out_layers.size( ); i++ )
						names[ i ] = layer_names[ out_layers[ i ] - 1 ];
				}
				return names;
			};

			m_net.setInput( m_blob );
			m_net.forward( m_output, out_names( m_net ) );

			return !!( m_output.size( ) > 1 );
		}

		std::int8_t post_process( ) {
			if ( m_output.size( ) <= 1 ) {
				std::cerr << "[dnn] bad output size\n";
				return 0;
			}

			std::vector< std::float_t >confidences;
			std::vector< std::int32_t >class_ids;
			std::vector< cv::Rect >boxes;

			const auto calc_box = [&](
				const cv::Mat cv_mat,
				const std::float_t hit_min,
				const std::float_t iou_max
			) {
				auto data{ reinterpret_cast< std::float_t* >( cv_mat.data ) };
				if ( !data ) {
					std::cerr << "[dnn] bad material data\n";
					return 0;
				}

				for ( std::size_t i{}; i < cv_mat.rows; i++, data += cv_mat.cols ) {

					cv::Point class_id{};
					std::double_t confidence{};
					cv::minMaxLoc( cv_mat.row( i ).colRange( 5, cv_mat.cols ), 0, &confidence, 0, &class_id );

					if ( confidence >= m_config.m_hit_min ) {
						auto size_x{ static_cast< std::int32_t >( data[ 2 ] * m_image.cols ) };
						auto size_y{ static_cast< std::int32_t >( data[ 3 ] * m_image.rows ) };

						boxes.push_back( cv::Rect{ 
							static_cast< std::int32_t >( data[ 0 ] * m_image.cols ) - size_x / 2, 
							static_cast< std::int32_t >( data[ 1 ] * m_image.rows ) - size_y / 2, 
							size_x, 
							size_y 
						} );

						class_ids.push_back( class_id.x );
						confidences.push_back( static_cast< std::float_t >( confidence ) );
					}
				}
				return 1;
			};

			const auto draw_box = [&](
				const std::int32_t class_id,
				const std::float_t confidence,
				const cv::Rect box,
				const std::int32_t font_face = cv::HersheyFonts::FONT_HERSHEY_SIMPLEX,
				const std::int32_t font_size = 1
			) {
				cv::rectangle( m_image, box, cv::Scalar{ 255, 255, 255 } );
				if ( m_names.empty( ) ) {
					std::cout << "[dnn] no names parsed... skipping\n";
					return 1;
				}

				if ( !m_names.empty( ) ) {
					auto conf_label{ cv::format( "%s, %.2f", m_names[ class_id ].data( ), confidence ) };
					if ( conf_label.empty( ) ) {
						std::cerr << "[dnn] failed to format label\n";
						return 0;
					}
					std::cout << "[dnn] " << m_names[ class_id ] << " " << confidence << "\n";
					cv::putText( m_image, conf_label.data( ), 
						cv::Point{ box.x, box.y }, font_face, font_size, cv::Scalar{ 255, 255, 255 } );
				}
				return 1;
			};

			for ( auto& it : m_output )
				calc_box( it, m_config.m_hit_min, m_config.m_iou_max );

			std::vector< std::int32_t >indices;
			cv::dnn::NMSBoxes( boxes, confidences, m_config.m_hit_min, m_config.m_iou_max, indices );

			for ( auto& it : indices )
				draw_box( class_ids[ it ], confidences[ it ], boxes[ it ] );

			return 1;
		}

		std::vector< std::string >m_names;
		std::vector< cv::Mat >m_output;
		cv::Mat m_image;
		cv::Mat m_blob;
		cv::dnn::Net m_net;
		model_conf_t m_config;
	};
}

std::int32_t main( ) {
	cv::utils::logging::setLogLevel( cv::utils::logging::LOG_LEVEL_WARNING );

	sdk::model_conf_t config{
		.m_model_path = "net/models/yolov3.weights",
		.m_names_path = "net/names/coco.names",
		.m_blob_x = 416,
		.m_blob_y = 416,
		.m_hit_min = 0.3f,
		.m_iou_max = 0.4f,
		.m_model_conf = "net/config/yolov3.cfg",
		.m_model_type = "Darknet"
	};

	sdk::model_t model{ config };
	if ( model.m_net.empty( ) )
		return 0;

	std::string files[] = {
		"shibuya_crossing.jpg",
		"people_walking.jpg",
		"giraffe.jpg",
		"horses.jpg", 
		"scream.jpg",
		"person.jpg",
		"eagle.jpg",
		"kite.jpg",
		"dog.jpg"
	};
	
	for ( auto& it : files ) {
		model.detect_image( "net/" + it );
		model.post_process( );

		cv::imshow( it, model.m_image );
	}
	cv::waitKey( 0 );
}
