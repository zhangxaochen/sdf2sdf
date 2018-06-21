#include <opencv/ml.h>
#include <opencv/highgui.h>
#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/pcl_visualizer.h>

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

#include "bilateral.hpp"

//syn:
// #define FX 570.3999633789062
// #define FY 570.3999633789062
// #define CX 320.0
// #define CY 240.0
//kinect:
#define FX_k 538.925221
#define FY_k 538.925221
#define CX_k 316.473843
#define CY_k 243.262082

// pcl::PointXYZ vxlDbg(47, 107, 258);
pcl::PointXYZ vxlDbg(73,129,366);

double FX = FX_k, FY = FY_k,
       CX = CX_k, CY = CY_k;

#define DELTA 0.005
double BETA = 0.5;
//expected object thickness
#define ETA 0.01

float SIDE_LENGTH = 0.002; //4mm; if 2mm, time cost inc cubic

void load_param_file(std::string filename)
{
    FILE *f = fopen(filename.c_str(), "r");
    if (f != NULL)
    {
        char buffer[1024];
        while (fgets(buffer, 1024, f) != NULL)
        {
            if (strlen(buffer) > 0 && buffer[0] != '#')
            {
                sscanf(buffer, "%lf", &FX);
                fgets(buffer, 1024, f);
                sscanf(buffer, "%lf", &FY);
                fgets(buffer, 1024, f);
                sscanf(buffer, "%lf", &CX);
                fgets(buffer, 1024, f);
                sscanf(buffer, "%lf", &CY);
                fgets(buffer, 1024, f);
                // sscanf(buffer, "%lf", &ICP_trunc_);
                // fgets(buffer, 1024, f);
                // sscanf(buffer, "%lf", &integration_trunc_);
            }
        }
        fclose(f);
        // PCL_WARN("Camera model set to (fx, fy, cx, cy, icp_trunc, int_trunc):\n\t%.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", fx_, fy_, cx_, cy_, ICP_trunc_, integration_trunc_);
        PCL_WARN("Camera model set to (fx, fy, cx, cy):\n\t%.2f, %.2f, %.2f, %.2f\n", FX, FY, CX, CY);
    }
}//load_param_file

cv::Mat load_exr_depth(std::string filename, bool isExr = true)
{
    // load the image
    cv::Mat depth_map = cv::imread( filename, -1 );
    if(isExr){
        // printf("depth_map.type(), depth_map.depth(): %d, %d\n", depth_map.type(), depth_map.depth()); //21, 5, i.e., 32fc3
        cv::cvtColor(depth_map, depth_map, CV_RGB2GRAY);
        // printf("depth_map.type(), depth_map.depth(): %d, %d\n", depth_map.type(), depth_map.depth()); //5, 5

        cv::Mat tmp = depth_map.clone();
        bilateral_filter<float>(tmp, depth_map);
    }
    else{

        cv::Mat tmp = depth_map.clone();
        // cv::bilateralFilter(tmp, depth_map, 3, 4, 4);
        bilateral_filter<ushort>(tmp, depth_map);
    }

    // convert to meters
    depth_map.convertTo( depth_map, CV_32FC1, 0.001 );

    return depth_map;
}//load_exr_depth

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
            if (z != 0 && !std::isinf(z))
            {
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

    printf("After padding......\n");
    printf("pointLL(%f,%f,%f)\n",pointll.x,pointll.y,pointll.z);
    printf("pointUR(%f,%f,%f)\n",pointur.x,pointur.y,pointur.z);
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
    printf("vxlDbg.xyz: %f, %f, %f\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);

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
                    bool doDbgPrint = false;
                    if (int(vxlDbg.x) == i && int(vxlDbg.y) == j && int(vxlDbg.z) == k){
                        doDbgPrint = true;
                        // printf("+++++++++++++++doDbgPrint\n");
                    }

                    pcl::PointXYZ intPoint(i, j, k);
                    pcl::PointXYZ floatPoint = getVoxelCenterByVoxel(intPoint, pointll, SIDE_LENGTH);
                    myPoint voxel(floatPoint.x, floatPoint.y, floatPoint.z);
                    double PhiRef = PhisFunc(FX, FY, CX, CY, depth_map_ref,
                                             voxel, initial_twist,
                                             DELTA, ETA, weight_ref, false);
                    if (doDbgPrint)
                        printf("PhiRef, weight_ref: %f, %f; ", PhiRef, weight_ref);
                    if (PhiRef < -1 || weight_ref == 0) continue;

                    double PhiTar = PhisFunc(FX, FY, CX, CY, depth_map_tar,
                                             voxel, initial_twist,
                                             DELTA, ETA, weight_tar, true);
                    if (doDbgPrint)
                        printf("PhiTar, weight_tar: %f, %f; \n", PhiTar, weight_tar);

                    if (PhiTar < -1 || PhiTar == PhiRef || weight_tar == 0) continue;

                    // if (doDbgPrint)
                    //     printf("phi1/2, w1/2: %f, %f, %f, %f\n", PhiRef, PhiTar, weight_ref, weight_tar);

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
                    if(doDbgPrint)
                        printf("sdf-err: %f\n", 0.5 * (PhiRef * weight_ref - PhiTar * weight_tar) *
                            (PhiRef * weight_ref - PhiTar * weight_tar));
                            
                    if ((PhiRef * weight_ref - PhiTar * weight_tar) *
                        (PhiRef * weight_ref - PhiTar * weight_tar) != 0)
                    error += 0.5 * (PhiRef * weight_ref - PhiTar * weight_tar) *
                             (PhiRef * weight_ref - PhiTar * weight_tar);
                }
            }
        }
        if(iter < 1)
            printf("vxl_valid_cnt: %d\n", vxl_valid_cnt);

#if 1

        Eigen::Matrix<double, 6, 1> inter_twist = A.inverse() * b;
        // cout << "A, b:" << A << endl
        //      << b << endl
        //      << "A.det: " << A.determinant() << endl
        //      << "A'*b: " << inter_twist.transpose() << endl;

        initial_twist += BETA * (inter_twist - initial_twist);

#else
        //zc: try direct GD myself
        initial_twist -= 1e-7 * delta_twist;
        cout << "delta_twist:" << delta_twist.transpose() << endl
             << "A:" << A << endl;
#endif

        // printf("%d th, error = %lf, vxl_valid_cnt: %d, avg-err: %lf\n", iter, error, vxl_valid_cnt, error / vxl_valid_cnt);
        // printf("twist is : ");
        // for (int l = 0; l < 6; l ++)
        //     printf("%f \n",initial_twist(l,0));
    }//for-iter
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

const std::string pt_picked_str = "MousePickedPoint";

void pointPickingCallback(const pcl::visualization::PointPickingEvent &event, void *cookie){
    if (event.getPointIndex() == -1)
        return;

    pcl::PointXYZ pt_picked;
    event.getPoint(pt_picked.x, pt_picked.y, pt_picked.z);
    cout << pt_picked << endl;
    // Vector3f cell_size = this->kinfu_->volume().getVoxelSize();
    pcl::visualization::PCLVisualizer *viewer = (pcl::visualization::PCLVisualizer *)(cookie);
    viewer->removeShape(pt_picked_str);
    viewer->addSphere(pt_picked, SIDE_LENGTH/2, 1,0,1, pt_picked_str);

    // vxlDbg.x = pt_picked.x;
    // vxlDbg.y = pt_picked.y;
    // vxlDbg.z = pt_picked.z;
    pcl::PointXYZ pt_ll(-0.148632, -0.250084, -0.004000); //kinect_bunny
    vxlDbg = getVoxel(pt_picked, pt_ll, SIDE_LENGTH);
    printf("@pointPickingCallback-vxlDbg.xyz: %f, %f, %f\n", vxlDbg.x, vxlDbg.y, vxlDbg.z);
}//pointPickingCallback

int main(int argc, char *argv[])
{
#ifdef NDEBUG
    printf(">>>>>>Release\n");
#else
    printf(">>>>>>Debug\n");
#endif

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

    // Eigen::AngleAxisf tt = Eigen::AngleAxisf(0.1, Eigen::Vector3f::UnitZ ());
    // cout<<tt.axis()<<endl;
    // Eigen::Matrix<float, 6, 1> xi_curr;
    // Eigen::Matrix<float, 3, 1> t3;
    // t3<<1,2,3;
    // // xi_curr=t3;
    // xi_curr << t3, t3;
    // cout<<t3<<','<<xi_curr<<endl;
    // cout<<t3(0)<<endl;
    // t3<<xi_curr;
    // // cout<<tt(2)<<endl;

    // Eigen::Vector3d t(1,0,0);           // 沿X轴平移1  
    // Sophus::SE3 SE3_Rt(R, t);           // 从R,t构造SE(3)  
    // Sophus::SE3 SE3_qt(q,t);            // 从q,t构造SE(3)  
    // cout<<"SE3 from R,t= "<<endl<<SE3_Rt<<endl;  
    // cout<<"SE3 from q,t= "<<endl<<SE3_qt<<endl;  
    // // 李代数se(3) 是一个六维向量，方便起见先typedef一下  
    // typedef Eigen::Matrix<double,6,1> Vector6d;// Vector6d指代　Eigen::Matrix<double,6,1>  
    // Vector6d se3 = SE3_Rt.log();  
    // cout<<"se3 = "<<se3.transpose()<<endl;  
    // // 观察输出，会发现在Sophus中，se(3)的平移在前，旋转在后.  
    // // 同样的，有hat和vee两个算符  
    // cout<<"se3 hat = "<<endl<<Sophus::SE3::hat(se3)<<endl;  
    // cout<<"se3 hat vee = "<<Sophus::SE3::vee( Sophus::SE3::hat(se3) ).transpose()<<endl;  
      
    // return 0;

    printf("-----------FX, FY, CX, CY: %f, %f, %f, %f\n", FX, FY, CX, CY);

    std::string eval_folder = "";
    pc::parse_argument(argc, argv, "-eval", eval_folder);

    bool use_omask = pc::find_switch(argc, argv, "-om");
    bool use_tmask = pc::find_switch(argc, argv, "-tm");
    bool use_exr = pc::find_switch(argc, argv, "-exr");
    if (eval_folder.find("Synthetic") != std::string::npos)
        use_exr = true;
    if (use_exr) //*.exr has no omask.png accompanied
        use_omask = use_tmask = false;
    PCL_WARN("\tuse_omask: %s, use_tmask: %s, use_exr: %s\n", 
    use_omask ? "TTT" : "FFF", use_tmask ? "TTT" : "FFF", use_exr ? "TTT" : "FFF");

    std::string camera_file;
    if(pc::parse_argument(argc, argv, "-param", camera_file) > 0)
        load_param_file(camera_file);
    
#if 0 //data sequences

    //init frame 0, set as Identity
    Vector6d twist0 = Vector6d::Zero();
    SE3d se_0i = SE3d::exp(twist0); //T0->Ti, total transformation
    SE3d se_i_0 = se_0i.inverse(); //i->0

    const Eigen::Quaterniond q_i = se_i_0.unit_quaternion(); //frame0:=I
    Vector3d t_i = se_i_0.translation();

    ofstream fout("s2s_poses.csv");
    //t.xyz+q.wxyz
    fout << t_i.x() << ',' << t_i.y() << ',' << t_i.z() << ',' << q_i.w() << ',' << q_i.x() << ',' << q_i.y() << ',' << q_i.z() << endl;

    // std::string path_dat = "Synthetic_Kenny_Circle/";
    // path_dat = "Kinect_Bunny_Turntable/";
    
    // std::string pat_dmap = path_dat + "depth_*.exr"; //for syn-...
    // pat_dmap = path_dat + "depth_*.png"; //for kinect-...
    // std::string pat_omsk = path_dat + "omask_*.png";
    std::string pat_dmap, pat_omsk, pat_tmsk;
    if(use_exr){
        pat_dmap = eval_folder + "/depth_*.exr";
    }
    else{ //should be png
        pat_dmap = eval_folder + "/depth_*.png";
        if(use_omask)
            pat_omsk = eval_folder + "/omask_*.png";
        if (use_tmask)
            pat_tmsk = eval_folder + "/tmask_*.png";
    }
    printf("pat_dmap: %s\n, pat_omsk: %s\n", pat_dmap.c_str(), pat_omsk.c_str());

    glob_t glob_dmap, glob_omsk, glob_tmsk;
    glob(pat_dmap.c_str(), GLOB_TILDE, NULL, &glob_dmap);
    glob(pat_omsk.c_str(), GLOB_TILDE, NULL, &glob_omsk);
    glob(pat_tmsk.c_str(), GLOB_TILDE, NULL, &glob_tmsk);

    printf("dmap-cnt: %lu, omsk-cnt: %lu\n", glob_dmap.gl_pathc, glob_omsk.gl_pathc);

    const int FRAME_INTERV = 1;
    for (size_t i = 0; i + FRAME_INTERV < glob_dmap.gl_pathc; i += FRAME_INTERV)
    // for (size_t i = 0; i + FRAME_INTERV < 40; i += FRAME_INTERV)
    {
        // printf("%s\n", glob_dmap.gl_pathv[i]);
        printf("-----%lu, %s\n", i, glob_dmap.gl_pathv[i]);

        //get depth map
        cv::Mat depth_map_ref = load_exr_depth(glob_dmap.gl_pathv[i], use_exr);
        cv::Mat depth_map_tar = load_exr_depth(glob_dmap.gl_pathv[i + FRAME_INTERV], use_exr);
        if(use_omask)
        {
            cv::Mat omask = cv::imread(glob_omsk.gl_pathv[i], -1);
            if(use_tmask){
                cv::Mat tmask = cv::imread(glob_tmsk.gl_pathv[i], -1);
                omask = omask + tmask;
                // cv::imshow("omask+tmask", omask);
                // cv::waitKey(0);
            }
            depth_map_ref.setTo(0, omask == 0);

            omask = cv::imread(glob_omsk.gl_pathv[i + FRAME_INTERV], -1);
            if(use_tmask){
                cv::Mat tmask = cv::imread(glob_tmsk.gl_pathv[i] + FRAME_INTERV, -1);
                omask = omask + tmask;
            }
            depth_map_tar.setTo(0, omask == 0);

            //dilate ref img, to de-noise isolated point
            int ksz = 2;
            cv::Mat krnl = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2*ksz+1, 2*ksz+1));
            // cv::morphologyEx(depth_map_ref, depth_map_ref, cv::MORPH_OPEN, krnl);

            // cv::Mat tmp = depth_map_ref.clone();
            // cv::bilateralFilter(tmp, depth_map_ref, 3, 4, 4);
            // tmp = depth_map_tar.clone();
            // cv::bilateralFilter(tmp, depth_map_tar, 3, 4, 4);
        }

        //i->i+1
        Sophus::Vector6d twist = get_twist(depth_map_ref, depth_map_tar);
        cout << "twist: " << twist.transpose() << endl;

        Sophus::SE3d se_ii1 = Sophus::SE3d::exp(twist); //i->(i+1)

        // se_0i *= se_ii1;             //0->i->(i+1) //wrong, because right mutiply
        // se_i_0 = se_0i.inverse(); //i->0, c2g
        se_i_0 *= se_ii1.inverse(); //0<-i<-(i+1), c2g

        // when save to txt, use i->0, camera2global, c2g
        const Eigen::Quaterniond q_i = se_i_0.unit_quaternion();
        t_i = se_i_0.translation();
        fout << t_i.x() << ',' << t_i.y() << ',' << t_i.z() << ',' << q_i.w() << ',' << q_i.x() << ',' << q_i.y() << ',' << q_i.z() << endl;
    }
    fout.close();
    globfree(&glob_dmap);

    printf(">>>fxy,cxy: %f, %f, %f, %f\n  delta,eta: %f, %f, beta,slen: %f, %f\n",
           FX, FY, CX, CY, DELTA, ETA, BETA, SIDE_LENGTH);
    printf("-----------DONE-----------\n");

    return 0;

#else
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr final_tar_point_cloud (new pcl::PointCloud<pcl::PointXYZ>);

    //get depth map
    // cv::Mat depth_map_ref = load_exr_depth("Synthetic_Kenny_Circle/depth_000000.exr");
    // cv::Mat depth_map_tar = load_exr_depth("Synthetic_Kenny_Circle/depth_000003.exr");
    cv::Mat depth_map_ref = load_exr_depth(eval_folder + "/depth_000000.png", false);
    cv::Mat depth_map_tar = load_exr_depth(eval_folder + "/depth_000003.png", false);

    if (use_omask)
    {
        cv::Mat omask = cv::imread(eval_folder + "/omask_000000.png", -1);
        // depth_map_ref = depth_map_ref[omask != 0];
        depth_map_ref.setTo(0, omask == 0);
        omask = cv::imread(eval_folder + "/omask_000003.png", -1);
        // depth_map_tar = depth_map_tar[omask != 0];
        depth_map_tar.setTo(0, omask == 0);
    }

    //get point cloud
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_ref,ref_point_cloud,0);
    DepthFrameToVertex(FX,FY,CX,CY,depth_map_tar,tar_point_cloud,0);

    Eigen::Matrix<double ,6,1> initial_twist = get_twist(depth_map_ref, depth_map_tar, ref_point_cloud, tar_point_cloud);
    cout << "twist: " << initial_twist.transpose() << endl;

    //convert the target to reference
    //get the reverse of reference position
    Sophus::SE3d se = Sophus::SE3d::exp(initial_twist);

    // i6->i0
    // Sophus::SE3d se(Eigen::Quaterniond(0.9962187260,-0.0008659845,-0.0742914162,-0.0450369721), {0.109307,-0.00638849,0.00853807}); //GT
    // Sophus::SE3d se(Eigen::Quaterniond(0.997138,0.00165619,-0.0486698,-0.0578235), {0.0741724,-0.00285784,0.00362768}); //test
    // i1->i0
    // Sophus::SE3d se(Eigen::Quaterniond(0.9999389981,-0.0002195521,-0.0095153305,-0.0056299934), {0.0141771000,-0.0004258450,0.0001466270}); //GT
    // Sophus::SE3d se(Eigen::Quaterniond(0.997138,0.00165619,-0.0486698,-0.0578235), {-0.00543874,-1.5327e-05,2.59581e-05}); //test, BAD
    // Sophus::SE3d se(Eigen::Quaterniond(0.999981, 2.98226e-05, 0.00503015, -0.00365883), {-0.00543874,-1.5327e-05,2.59581e-05}); //test, GOOD
    // se = se.inverse(); //i0->ix, keep consistent

    Eigen::Matrix<double, 4, 4> inverse_homogenous = (se.inverse()).matrix();
    cout << "twist-inverse: " << se.inverse().log().transpose() << endl;

    Eigen::Quaterniond q_inv = se.inverse().unit_quaternion();
    cout << "inv:-> t.xyz, q.wxyz: " << se.inverse().translation().transpose() << ", " << q_inv.w() << ", " << q_inv.x() << ", " << q_inv.y() << ", " << q_inv.z() << ", " << endl;

    pcl::transformPointCloud (*tar_point_cloud, *final_tar_point_cloud, inverse_homogenous);
    cout << "inverse_homogenous:\n"
         << inverse_homogenous << endl;

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

    viewer.registerPointPickingCallback(&pointPickingCallback, (void*)&viewer);

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
