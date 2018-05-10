//zc: 2018-5-9 18:03:01
#ifndef _BILATERAL_H_
#define _BILATERAL_H_

#include <limits>
#include <cmath>

#include <opencv2/opencv.hpp>

//@param[in] src, input dmap of type *DTYPE*
//@param[in|out] dst, filtered output dmap of type *DTYPE*
template<typename DTYPE>
void bilateral_filter(cv::Mat src, cv::Mat dst){
    using namespace std;

    const float sigma_color = 30;  //in mm
    const float sigma_space = 4.5; // in pixels

    float sigma_space2_inv_half = 0.5 / (sigma_space * sigma_space),
          sigma_color2_inv_half = 0.5 / (sigma_color * sigma_color);

    // const int R = 6;
    const int R = static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    for (int x = 0; x < src.cols; ++x)
    {
        for (int y = 0; y < src.rows; ++y)
        {

            // int value = src.at<ushort>(y, x);
            DTYPE value = src.at<DTYPE>(y, x);
            if (std::isinf(value))
            {
                dst.at<DTYPE>(y, x) = value;
                continue;
            }

            int tx = min(x - D / 2 + D, src.cols - 1);
            int ty = min(y - D / 2 + D, src.rows - 1);

            float sum1 = 0;
            float sum2 = 0;

            for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
            {
                for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
                {
                    // int tmp = src.at<ushort>(cy, cx);
                    DTYPE tmp = src.at<DTYPE>(cy, cx);
                    if (std::isinf(tmp))
                        continue;

                    float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    float color2 = (value - tmp) * (value - tmp);

                    float weight = expf(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

                    sum1 += tmp * weight;
                    sum2 += weight;
                }
            }

            // int res = __float2int_rn(sum1 / sum2);
            // int res = int(sum1 / sum2 + 0.5); //round-nearest
            DTYPE res = (DTYPE)(sum1 / sum2);
            dst.at<DTYPE>(y, x) = max((DTYPE)0, min(res, (DTYPE)numeric_limits<DTYPE>::max()));
        }
    }
} //bilateral_filter


#endif // _BILATERAL_H_
