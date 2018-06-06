#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <ctime>

const float * SCORES_REF = nullptr;
bool cmp_func(int i, int j){
    return SCORES_REF[i] > SCORES_REF[j];
}

// compute iou for two boxs
float compute_iou(const float * box1,
                  const float * box2){
    if (box1[2] <= box2[0] || box1[0] >= box2[2] ||
        box1[3] <= box2[1] || box1[1] >= box2[3])
        return 0;

    float s1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float s2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float si = (std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]))
               * (std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]));
    float su = s1 + s2 - si;

    if (su > 0)
        return si / su;
    else{
        std::cerr<<"[ERROR] Got a box with zero area."<<std::endl;
        exit(1);
    }
}

// decode boxes
float DEFAULT_BOX_SCALARS[4] = {10.f, 10.f, 5.f, 5.f};
void decode(float * decoded,
            const float * box_encodings,                 // [num_anchors * 4] of float32
            int num_filtered_anchors,
            const int * indices_filtered_anchors,        // [num_anchors] of int32
            const float * anchors,                       // [num_anchors * 4] of float32
            const float box_scalars[4] = DEFAULT_BOX_SCALARS){
    for(int i = 0; i < num_filtered_anchors; i++){
        int index = indices_filtered_anchors[i];

        float ty = box_encodings[index * 4 + 0];
        float tx = box_encodings[index * 4 + 1];
        float th = box_encodings[index * 4 + 2];
        float tw = box_encodings[index * 4 + 3];

        float ycenter_a = anchors[index * 4 + 0];
        float xcenter_a = anchors[index * 4 + 1];
        float ha = anchors[index * 4 + 2];
        float wa = anchors[index * 4 + 3];

        ty = ty / box_scalars[0];
        tx = tx / box_scalars[1];
        th = th / box_scalars[2];
        tw = tw / box_scalars[3];

        float h = exp(th) * ha;
        float w = exp(tw) * wa;
        float ycenter = ty * ha + ycenter_a;
        float xcenter = tx * wa + xcenter_a;

        decoded[i * 4 + 0] = ycenter - h / 2.f;
        decoded[i * 4 + 1] = xcenter - w / 2.f;
        decoded[i * 4 + 2] = ycenter + h / 2.f;
        decoded[i * 4 + 3] = xcenter + w / 2.f;
    }
}

void anchor_generate(float * anchors,
                int num_anchors,
                const std::string & file_path = ""){
//    std::clock_t t1 = std::clock();

    if(!file_path.empty()){
        std::ifstream anchor_file(file_path, std::ios::binary);
        int count = 0;
        while (anchor_file.read(reinterpret_cast<char*>(anchors + count), sizeof(float)))
            count++;
        anchor_file.close();
    }
    else{
        std::cerr<<"[ERROR] Anchor file_path not provided. But anchor generation code not implemented yet."<<std::endl;
    }

//    std::clock_t t2 = std::clock();
//    std::cout << "Anchor reading time: " << (t2 - t1) / float(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
}

// post-processing of detection, mainly decoding (generate anchors
// if not existed) and non-max-suppression (multi class version).
void post_det(float * &pred_boxes,
              float * &pred_scores,
              int * &pred_classes,
              int &num_pred_boxes,

              const float * box_encodings,          // [num_anchors * 4] of float32
              const float * class_scores,           // [num_classes * num_anchors] of float32
              const float * anchors,      // generate anchors if NULL

              float iou_thres = 0.6,
              float score_thres = 0.5,
              int num_anchors = 1917,
              int num_classes = 91,
              int max_preds = 100,
              int bg_class_index = 0,
              bool just_do_person = false){

    auto * decoded = new float[max_preds * 4];
    for(int c = 0; c < num_classes; c++){
        if(c == bg_class_index)
            continue;
        if(just_do_person && c != 1)
            continue;

        // variables to track
        int num_filtered_anchors = 0;
        std::vector<int> indices_filtered_anchors;
        auto scores = class_scores + c * num_anchors;

        // remove indices with scores lower than score_thres
        for(int a = 0; a < num_anchors; a++){
            if(scores[a] >= score_thres){
                num_filtered_anchors++;
                indices_filtered_anchors.push_back(a);
            }
        }

        // sort
        SCORES_REF = scores;
        std::sort(indices_filtered_anchors.begin(), indices_filtered_anchors.end(), cmp_func);
        num_filtered_anchors = std::min(num_filtered_anchors, max_preds);

        // decode boxes
        decode(decoded, box_encodings,
               num_filtered_anchors,
               indices_filtered_anchors.data(),
               anchors);

        // do non max suppression
        int num_previous_accpted = num_pred_boxes;
        int num_accepted = 0;
        for(int i = 0; i < num_filtered_anchors; i++){
            int a = indices_filtered_anchors[i];
            float * box1 = decoded + i * 4;

            bool accepted = true;
            for(int j = 0; j < num_accepted; j++){
                float * box2 = pred_boxes + (num_previous_accpted + j) * 4;
                float iou = compute_iou(box1, box2);
                if (iou > iou_thres){
                    accepted = false;
                    break;
                }
            }

            if (accepted){
                memcpy(pred_boxes + num_pred_boxes * 4,
                       decoded + i * 4,
                       4 * sizeof(float));
                pred_scores[num_pred_boxes] = scores[a];
                pred_classes[num_pred_boxes] = c;
                num_accepted++;
                num_pred_boxes++;
            }
        }
    }
    delete[] decoded;

    // Remember to release pred_boxes, pred_scores, pred_classes, anchors, and decoded
}

int main() {
    // init
    float iou_thres = 0.6;
    float score_thres = 0.5;
    int num_anchors = 1917;
    int num_classes = 91;
    int max_preds = 100;
    int bg_class_index = 0;
    bool just_do_person = true;

    int num_valid_classes = just_do_person ? 2 : num_classes;
    auto pred_boxes = new float[max_preds * 4 * num_valid_classes];
    auto pred_scores = new float[max_preds * num_valid_classes];
    auto pred_classes = new int[max_preds * num_valid_classes];
    int num_pred_boxes = 0;

    // read data
    auto box_encodings = new float[num_anchors * 4];
    auto class_scores = new float[num_valid_classes * num_anchors];
    int count;

    std::ifstream box_file("box_encodings.bin", std::ios::binary);
    count = 0;
    while (box_file.read(reinterpret_cast<char*>(box_encodings + count), sizeof(float)))
        count++;
    box_file.close();

    std::ifstream score_file("class_scores.bin", std::ios::binary);
    count = 0;
    while (score_file.read(reinterpret_cast<char*>(class_scores + count), sizeof(float)))
        count++;
    score_file.close();

    // read or generate anchors
    std::string anchors_path = "anchors.bin";
    auto anchors = new float[num_anchors * 4];
    anchor_generate(anchors, num_anchors, anchors_path);

    // do postprocessing, including: decode boxes, clip to window (not done yet), and multi-class nms
    std::clock_t t1 = std::clock();

    post_det(pred_boxes, pred_scores, pred_classes, num_pred_boxes,
             box_encodings, class_scores, anchors,
             iou_thres, score_thres, num_anchors,
             num_classes, max_preds, bg_class_index,
             just_do_person);

    std::clock_t t2 = std::clock();

    // print outputs
    std::cout<<"num_pred_boxes: "<<num_pred_boxes<<std::endl;
    int width = 690;
    int height = 460;
    for(int i=0; i<num_pred_boxes; i++){
        std::cout<<"("<<pred_classes[i]<<", "
                 <<pred_scores[i]<<"), ["
                 <<pred_boxes[i*4+0] * height<<", "<<pred_boxes[i*4+1] * width<<", "
                 <<pred_boxes[i*4+2] * height<<", "<<pred_boxes[i*4+3] * width<<"]."
                 <<std::endl;
    }

    std::cout << "Total time: " << (t2 - t1) / float(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    // release memory
    delete[] pred_boxes;
    delete[] pred_scores;
    delete[] pred_classes;
    delete[] box_encodings;
    delete[] class_scores;
    delete[] anchors;

    return 0;
}