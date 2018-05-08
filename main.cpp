#include <opencv/ml.h>
#include <opencv/highgui.h>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <opencv/cvaux.h>
#include "PhiFuncGradients.h"
#include "sophus/se3.hpp"
#include <pcl/common/transforms.h>
//#include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
// #include <iostream>
#include <glob.h>
#include <fstream>

//syn:
#define FX 570.3999633789062
#define FY 570.39996337890622
#define CX 320.0
#define CY 240.0
//kinect:
#define FX 538.925221
#define FY 538.925221
#define CX 316.473843
#define CY 243.262082

#define DELTA 0.005
double BETA = 0.5;
//expected object thickness
#define ETA 0.01

float SIDE_LENGTH = 0.002; //4mm; if 2mm, time cost inc cubic

cv::Mat load_exr_depth(std::string filename, bool isExr = true)
{
    // load the image
    cv::Mat depth_map = cv::imread( filename, -1 );
    if(isExr)
        cv::cvtColor(depth_map, depth_map, CV_RGB2GRAY);

    // convert to meters
    depth_map.convertTo( depth_map, CV_32FC1, 0.001 );

    return depth_map;
}

cv::Mat mask_dmap(cv::Mat dmap, cv::Mat msk){
    // cv::Mat res = dmap[msk != 0];
    cv::Mat res = dmap.clone();
    res.setTo(0, msk == 0);
    return res;
}//mask_dmap

void DepthFrameToVertex(float fx,float fy,float cx,float cy,
                        cv::Mat &depth_image, pcl::PointCloud<pcl::PointXYZ>::Ptr target_pc, bool organized)
{
    float* pixel_ptr;
    target_pc->height = (uint32_t)depth_image.rows;
    target_pc->width = (uint32_t)depth_image.cols;
    target_pc->is_dense = false;
    target_pc->resize(target_pc->height * target_pc->width);
    for (int y = 0; y < depth_image.rows; ++y)
    {
        for  (int x = 0; x < depth_image.cols; ++x)
        {
            float z = depth_image.at<float>(y,x);
            target_pc->at(x, y).x = (x - cx) * z / fx;
            target_pc->at(x, y).y = (y - cy) * z / fy;
            target_pc->at(x, y).z = z;
            if (!std::isinf(z))
            {
                //printf("point(%f,%f,%f)\n",target_pc->at(y,x).x,target_pc->at(y,x).y,target_pc->at(y,x).z);
            }
            //++pixel_ptr;
        }
    }
}
pcl::PointXYZ getVoxel(pcl::PointXYZ &point, pcl::PointXYZ &c, float l)
{
    float x = (float) round((1/l) * (point.x - c.x)-(0.5));
    float y = (float) round((1/l) * (point.y - c.y)-(0.5));
    float z = (float) round((1/l) * (point.z - c.z)-(0.5));
    return pcl::PointXYZ(x,y,z);
}
pcl::PointXYZ getVoxelCenter(pcl::PointXYZ point, pcl::PointXYZ c, float l)
{
    pcl::PointXYZ voxel = getVoxel(point, c, l);
    float x = (float) (l * (voxel.x + 0.5) + c.x);
    float y = (float) (l * (voxel.y + 0.5) + c.y);
    float z = (float) (l * (voxel.z + 0.5) + c.z);
    return pcl::PointXYZ(x,y,z);
}
pcl::PointXYZ getVoxelCenterByVoxel(pcl::PointXYZ point, pcl::PointXYZ c, float l)
{
    float x = (float) (l * (point.x + 0.5) + c.x);
    float y = (float) (l * (point.y + 0.5) + c.y);
    float z = (float) (l * (point.z + 0.5) + c.z);
    return pcl::PointXYZ(x,y,z);
}

void getLowerLeftAndUpperRight(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
                               pcl::PointXYZ &pointll, pcl::PointXYZ &pointur)
{
    pointll.x = INFINITY; pointll.y = INFINITY; pointll.z = INFINITY;
    pointur.x = -INFINITY; pointur.y = -INFINITY; pointur.z = -INFINITY;
    for (int x = 0; x < point_cloud->height; ++x) {
        for (int y = 0; y < point_cloud->width; ++y) {
            pcl::PointXYZ point = point_cloud->at(y,x);
            if ((!std::isinf(point.x)) && (!std::isinf(-point.x)) && (point.x < pointll.x)) pointll.x = point.x;
            if ((!std::isinf(point.y)) && (!std::isinf(-point.y)) && (point.y < pointll.y)) pointll.y = point.y;
            if ((!std::isinf(point.z)) && (!std::isinf(-point.z)) && (point.z < pointll.z)) pointll.z = point.z;
            if ((!std::isinf(point.x)) && (!std::isinf(-point.x)) && (point.x > pointur.x)) pointur.x = point.x;
            if ((!std::isinf(point.y)) && (!std::isinf(-point.y)) && (point.y > pointur.y)) pointur.y = point.y;
            if ((!std::isinf(point.z)) && (!std::isinf(-point.z)) && (point.z > pointur.z)) pointur.z = point.z;
        }
    }
}

bool isValid(pcl::PointXYZ &point)
{
    return ((!std::isinf(point.x)) && (!std::isinf(point.y)) && (!std::isinf(point.z)) &&
            (!std::isinf(-point.x)) && (!std::isinf(-point.y)) && (!std::isinf(-point.z)));
}

Sophus::Vector6d get_twist(cv::Mat depth_map_ref, cv::Mat depth_map_tar, pcl::PointCloud<pcl::PointXYZ>::Ptr ref_point_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_point_cloud){
    //get the lower left and upper left points
    pcl::PointXYZ pointll, pointur;
    getLowerLeftAndUpperRight(ref_point_cloud,pointll,pointur);
    // printf("pointLL(%f,%f,%f)\n",pointll.x,pointll.y,pointll.z);
    // printf("pointUR(%f,%f,%f)\n",pointur.x,pointur.y,pointur.z);
    /* Result
     pointLL(-0.112570,0.013829,0.449250)
     pointUR(0.013296,0.124222,0.501750)
     */

    //padding...
    pointll.x -= 2 * SIDE_LENGTH; pointll.y -= 2 * SIDE_LENGTH; pointll.z -= 2 * SIDE_LENGTH;
    pointur.x += 2 * SIDE_LENGTH; pointur.y += 2 * SIDE_LENGTH; pointur.z += 2 * SIDE_LENGTH;

    // printf("After padding......\n");
    // printf("pointLL(%f,%f,%f)\n",pointll.x,pointll.y,pointll.z);
    // printf("pointUR(%f,%f,%f)\n",pointur.x,pointur.y,pointur.z);
    /* Result
     pointLL(-0.113570,0.012829,0.448250)
     pointUR(0.014296,0.125222,0.502750)
     */

    //initial twist
    Eigen::Matrix<double ,6,1> initial_twist = Eigen::MatrixXd::Zero(6,1);

    //optimization
    pcl::PointXYZ maxVoxel = getVoxel(pointur, pointll, SIDE_LENGTH);
    int max_x = (int) maxVoxel.x;
    int max_y = (int) maxVoxel.y;
    int max_z = (int) maxVoxel.z;
    printf("max_xyz: %d, %d, %d; total: %d\n", max_x, max_y, max_z, max_x * max_y * max_z);

    double weight_ref, weight_tar, weight_temp;

    clock_t begt = clock();
    for (int iter = 0; iter < 20; iter ++) {
        //if (iter == 40) BETA *= 0.1;
        //if (iter >= 40) SIDE_LENGTH = 0.002;
        Eigen::Matrix<double, 6, 6> A = Eigen::MatrixXd::Zero(6,6);
        Eigen::Matrix<double, 6, 1> b = Eigen::MatrixXd::Zero(6,1);

        //zc:
        Eigen::Matrix<double, 6, 1> delta_twist = Eigen::MatrixXd::Zero(6,1);

        int vxl_total_cnt = 0,
            vxl_valid_cnt = 0;

        double error = 0;
        for (int i = 0; i < max_x; i++) {
            for (int j = 0; j < max_y; j++) {
                for (int k = 0; k < max_z; k++) {
                    pcl::PointXYZ intPoint(i, j, k);
                    pcl::PointXYZ floatPoint = getVoxelCenterByVoxel(intPoint, pointll, SIDE_LENGTH);
                    myPoint voxel(floatPoint.x, floatPoint.y, floatPoint.z);
                    double PhiRef = PhisFunc(FX, FY, CX, CY, depth_map_ref,
                                             voxel, initial_twist,
                                             DELTA, ETA, weight_ref, false);
                    if (PhiRef < -1 || weight_ref == 0) continue;

                    double PhiTar = PhisFunc(FX, FY, CX, CY, depth_map_tar,
                                             voxel, initial_twist,
                                             DELTA, ETA, weight_tar, true);
                    if (PhiTar < -1 || PhiTar == PhiRef || weight_tar == 0) continue;

                    // if ((PhiRef < -1) || (PhiTar < -1)) continue;

                    // //zc: To speed it up
                    // if (weight_ref == 0 || weight_tar == 0 || PhiRef == PhiTar)
                    //     continue;
                    vxl_valid_cnt += 1;
                    
                    Eigen::Matrix<double, 1, 6> gradient = PhiFuncGradients(FX, FY, CX, CY, depth_map_tar,
                                                                            voxel, initial_twist,
                                                                            DELTA, ETA, SIDE_LENGTH, weight_temp);
                    A += gradient.transpose() * gradient;
                    b += (PhiRef -
                          PhiTar +
                          gradient * initial_twist) *
                         gradient.transpose();
                    //zc:
                    delta_twist += (PhiTar - PhiRef) * gradient.transpose();

                    if ((PhiRef * weight_ref - PhiTar * weight_tar) *
                        (PhiRef * weight_ref - PhiTar * weight_tar) != 0)
                    error += 0.5 * (PhiRef * weight_ref - PhiTar * weight_tar) *
                             (PhiRef * weight_ref - PhiTar * weight_tar);
                }
            }
        }
        if(iter < 1)
            printf("vxl_valid_cnt: %d\n", vxl_valid_cnt);

        Eigen::Matrix<double, 6, 1> inter_twist = A.inverse() * b;
        initial_twist += BETA * (inter_twist - initial_twist);

        //zc: try direct GD myself
        // initial_twist -= 1e-7 * delta_twist;
        // cout << "delta_twist:" << delta_twist.transpose() << endl
        //      << "A:" << A << endl;

        // printf("%d th, error = %lf, vxl_valid_cnt: %d, avg-err: %lf\n", iter, error, vxl_valid_cnt, error / vxl_valid_cnt);
        // printf("twist is : ");
        // for (int l = 0; l < 6; l ++)
        //     printf("%f \n",initial_twist(l,0));
    }
    printf("time-cost: %f\n", double(clock()-begt)/CLOCKS_PER_SEC);

    return initial_twist;
}//get_twist

Sophus::Vector6d get_twist(cv::Mat depth_map_ref, cv::Mat depth_map_tar){
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    //get point cloud
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_ref,ref_point_cloud,0);
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_tar,tar_point_cloud,0);

    return get_twist(depth_map_ref, depth_map_tar, ref_point_cloud, tar_point_cloud);
}//get_twist

int main(int argc, char *argv[])
{
    using namespace std;
    using namespace Sophus;
    namespace pc = pcl::console;

    // Vector6d tmp = Vector6d::Zero();
    // cout << tmp << endl;
    // SE3d se = SE3d::exp(tmp);
    // Eigen::Quaterniond q = se.unit_quaternion();

    // cout << se.matrix() << endl
    //      << q.w() << q.x() << q.y() << q.z() << endl;

    // return 0 ;

    printf("-----------FX, FY, CX, CY: %f, %f, %f, %f\n", FX, FY, CX, CY);

    std::string eval_folder = "";
    pc::parse_argument(argc, argv, "-eval", eval_folder);

    bool use_omask = pc::find_switch(argc, argv, "-om");
    bool use_exr = pc::find_switch(argc, argv, "-exr");
    if (eval_folder.find("Synthetic") != std::string::npos)
        use_exr = true;
    if (use_exr) //*.exr has no omask.png accompanied
        use_omask = false;

#if 10 //data sequences

    //init frame 0, set as Identity
    Vector6d twist0 = Vector6d::Zero();
    SE3d se_0i = SE3d::exp(twist0); //T0->Ti, total transformation
    SE3d se_0i_inv = se_0i.inverse();

    const Eigen::Quaterniond q_i = se_0i_inv.unit_quaternion(); //frame0:=I
    Vector3d t_i = se_0i_inv.translation();

    ofstream fout("s2s_poses.csv");
    //t.xyz+q.wxyz
    fout << t_i.x() << ',' << t_i.y() << ',' << t_i.z() << ',' << q_i.w() << ',' << q_i.x() << ',' << q_i.y() << ',' << q_i.z() << endl;

    // std::string path_dat = "Synthetic_Kenny_Circle/";
    // path_dat = "Kinect_Bunny_Turntable/";
    
    // std::string pat_dmap = path_dat + "depth_*.exr"; //for syn-...
    // pat_dmap = path_dat + "depth_*.png"; //for kinect-...
    // std::string pat_omsk = path_dat + "omask_*.png";
    std::string pat_dmap, pat_omsk;
    if(use_exr){
        pat_dmap = eval_folder + "/depth_*.exr";
    }
    else{ //should be png
        pat_dmap = eval_folder + "/depth_*.png";
        if(use_omask)
            pat_omsk = eval_folder + "/omask_*.png";
    }
    printf("pat_dmap: %s\n, pat_omsk: %s\n", pat_dmap.c_str(), pat_omsk.c_str());

    glob_t glob_dmap, glob_omsk;
    glob(pat_dmap.c_str(), GLOB_TILDE, NULL, &glob_dmap);
    glob(pat_omsk.c_str(), GLOB_TILDE, NULL, &glob_omsk);

    printf("dmap-cnt: %lu, omsk-cnt: %lu\n", glob_dmap.gl_pathc, glob_omsk.gl_pathc);

    const int FRAME_INTERV = 1;
    for (size_t i = 0; i + FRAME_INTERV < glob_dmap.gl_pathc; i += FRAME_INTERV)
    {
        // printf("%s\n", glob_dmap.gl_pathv[i]);
        printf("-----%lu, ", i);

        //get depth map
        cv::Mat depth_map_ref = load_exr_depth(glob_dmap.gl_pathv[i], use_exr);
        cv::Mat depth_map_tar = load_exr_depth(glob_dmap.gl_pathv[i + FRAME_INTERV], use_exr);
        if(use_omask)
        {
            cv::Mat omask = cv::imread(glob_omsk.gl_pathv[i], -1);
            // depth_map_ref = depth_map_ref[omask != 0];
            depth_map_ref.setTo(0, omask == 0);
            omask = cv::imread(glob_omsk.gl_pathv[i + FRAME_INTERV], -1);
            // depth_map_tar = depth_map_tar[omask != 0];
            depth_map_tar.setTo(0, omask == 0);

            //dilate ref img, to de-noise isolated point
            int ksz = 2;
            cv::Mat krnl = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*ksz+1, 2*ksz+1));
            // cv::morphologyEx(depth_map_ref, depth_map_ref, cv::MORPH_OPEN, krnl);

            // cv::bilateral
        }

        //i->i+1
        Sophus::Vector6d twist = get_twist(depth_map_ref, depth_map_tar);
        Sophus::SE3d se_ii1 = Sophus::SE3d::exp(twist); //i->(i+1)

        se_0i *= se_ii1;             //0->i->(i+1)
        se_0i_inv = se_0i.inverse(); //i->0, c2g

        // when save to txt, use i->0, camera2global, c2g
        const Eigen::Quaterniond q_i = se_0i_inv.unit_quaternion();
        t_i = se_0i_inv.translation();
        fout << t_i.x() << ',' << t_i.y() << ',' << t_i.z() << ',' << q_i.w() << ',' << q_i.x() << ',' << q_i.y() << ',' << q_i.z() << endl;
    }
    fout.close();
    globfree(&glob_dmap);
    
    printf("-----------DONE-----------\n");

    return 0;

#else
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_tar_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    //get depth map
    // cv::Mat depth_map_ref = load_exr_depth("Synthetic_Kenny_Circle/depth_000000.exr");
    // cv::Mat depth_map_tar = load_exr_depth("Synthetic_Kenny_Circle/depth_000003.exr");
    cv::Mat depth_map_ref = load_exr_depth("Kinect_Kenny_Turntable/depth_000000.png", false);
    cv::Mat depth_map_tar = load_exr_depth("Kinect_Kenny_Turntable/depth_000001.png", false);

    if (use_omask)
    {
        cv::Mat omask = cv::imread("Kinect_Kenny_Turntable/omask_000000.png", -1);
        // depth_map_ref = depth_map_ref[omask != 0];
        depth_map_ref.setTo(0, omask == 0);
        omask = cv::imread("Kinect_Kenny_Turntable/omask_000001.png", -1);
        // depth_map_tar = depth_map_tar[omask != 0];
        depth_map_tar.setTo(0, omask == 0);
    }

    //get point cloud
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_ref,ref_point_cloud,0);
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_tar,tar_point_cloud,0);

    Eigen::Matrix<double ,6,1> initial_twist = get_twist(depth_map_ref, depth_map_tar, ref_point_cloud, tar_point_cloud);
    
    //convert the target to reference
    //get the reverse of reference position
    Sophus::SE3d se = Sophus::SE3d::exp(initial_twist);
    Eigen::Matrix<double, 4, 4> inverse_homogenous = (se.inverse()).matrix();
    pcl::transformPointCloud (*tar_point_cloud, *final_tar_point_cloud, inverse_homogenous);

    // Visualization
    pcl::visualization::PCLVisualizer viewer ("Matrix transformation");

    // Define R,G,B colors for the point cloud
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (ref_point_cloud, 255, 255, 255);
    // We add the point cloud to the viewer and pass the color handler
    viewer.addPointCloud (ref_point_cloud, source_cloud_color_handler, "ref_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (final_tar_point_cloud, 230, 20, 20); // Red
    viewer.addPointCloud (final_tar_point_cloud, transformed_cloud_color_handler, "target_cloud");

    //viewer.addCoordinateSystem (1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ref_cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud");
    viewer.setPosition(800, 600); // Setting visualiser window position

    //+++++++++++++++zc: compare with before alignment @2018-04-23 00:11:44
    pcl::visualization::PCLVisualizer viewer2 ("without transformation");
    viewer2.addPointCloud (ref_point_cloud, source_cloud_color_handler, "ref_cloud");

    viewer2.addPointCloud (tar_point_cloud, transformed_cloud_color_handler, "target_cloud");

    //viewer2.addCoordinateSystem (1.0, "cloud", 0);
    viewer2.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
    viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ref_cloud");
    viewer2.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target_cloud");
    viewer2.setPosition(800, 00); // Setting visualiser window position


    while (!viewer.wasStopped () && !viewer2.wasStopped()) { // Display the visualiser until 'q' key is pressed
        viewer.spinOnce ();
    }


    /*
    //Visualization
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (target_point_cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }
    */
#endif
}
