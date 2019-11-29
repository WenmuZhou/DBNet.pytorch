//
//  pse
//  Created by zhoujun on 11/9/19.
//  Copyright © 2019年 zhoujun. All rights reserved.
//
#include <queue>
#include <math.h>
#include <map>
#include <algorithm>
#include <vector>
#include "include/pybind11/pybind11.h"
#include "include/pybind11/numpy.h"
#include "include/pybind11/stl.h"
#include "include/pybind11/stl_bind.h"

namespace py = pybind11;


namespace pan{
    std::map<int,std::vector<float>> get_points(
    py::array_t<int32_t, py::array::c_style> label_map,
    py::array_t<float, py::array::c_style> score_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto pbuf_score_map = score_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        auto ptr_score_map = static_cast<float *>(pbuf_score_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::map<int,std::vector<float>> point_dict;
        std::vector<std::vector<float>> point_vector;
        for(int i=0;i<label_num;i++)
        {
            std::vector<float> point;
            point.push_back(0);
            point.push_back(0);
            point_vector.push_back(point);
        }
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            auto p_score_map = ptr_score_map + i*w;
            for(int j = 0; j<w; j++)
            {
                int32_t label = p_label_map[j];
                if(label==0)
                {
                    continue;
                }
                float score = p_score_map[j];
                point_vector[label][0] += score;
                point_vector[label][1] += 1;
                point_vector[label].push_back(j);
                point_vector[label].push_back(i);
            }
        }
        for(int i=0;i<label_num;i++)
        {
            if(point_vector[i].size() > 2)
            {
                point_vector[i][0] /= point_vector[i][1];
                point_dict[i] = point_vector[i];
            }
        }
        return point_dict;
    }
    std::vector<int> get_num(
    py::array_t<int32_t, py::array::c_style> label_map,
    int label_num)
    {
        auto pbuf_label_map = label_map.request();
        auto ptr_label_map = static_cast<int32_t *>(pbuf_label_map.ptr);
        int h = pbuf_label_map.shape[0];
        int w = pbuf_label_map.shape[1];

        std::vector<int> point_vector;
        for(int i=0;i<label_num;i++)
        {
            point_vector.push_back(0);
        }
        for (int i = 0; i<h; i++)
        {
            auto p_label_map = ptr_label_map + i*w;
            for(int j = 0; j<w; j++)
            {
                int32_t label = p_label_map[j];
                if(label==0)
                {
                    continue;
                }
                point_vector[label] += 1;
            }
        }
        return point_vector;
    }
}

PYBIND11_MODULE(pse, m){
    m.def("get_points", &pan::get_points, " re-implementation pse algorithm(cpp)", py::arg("label_map"), py::arg("score_map"), py::arg("label_num"));
    m.def("get_num", &pan::get_num, " re-implementation pse algorithm(cpp)", py::arg("label_map"), py::arg("label_num"));
}

