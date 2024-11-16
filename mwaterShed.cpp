
#include <vector>
#include "mwatershed.h"
using namespace cv;

namespace MWS {
    struct WSNode
    {
        int next;
        int mask_ofs;
        int img_ofs;
    };

    // Queue for WSNodes
    struct WSQueue
    {
        WSQueue() { first = last = 0; }
        int first, last;
    };


    static int allocWSNodes(std::vector<WSNode>& storage)
    {
        int sz = (int)storage.size();
        int newsz = MAX(128, sz * 3 / 2);

        storage.resize(newsz);
        if (sz == 0)
        {
            storage[0].next = 0;
            sz = 1;
        }
        for (int i = sz; i < newsz - 1; i++)
            storage[i].next = i + 1;
        storage[newsz - 1].next = 0;
        return sz;
    }

    void printBuf(cv::Mat src) {
        /*for (size_t i = 0; i < src.rows; i++)
        {
            for (size_t j = 0; j < src.cols; j++) {
                int val = (int)src.at<int>(i, j);
                std::string c = val == -1 ? "a" : (val == -2 ? "b" : std::to_string(int(val)));
                std::cout << c;
            }std::cout << std::endl;
        }std::cout << std::endl;*/

        namedWindow("mask", 2);
        cv::Mat src_8u = (src + 2) * 45;
        src_8u.convertTo(src_8u, CV_8U);
        imshow("mask", src_8u);
        waitKey();
    }

    void printVec(std::vector<WSNode>storage)
    {
        for (auto node : storage) {
            std::cout << "maskIdx: "<<node.mask_ofs << " nodenext: " << node.next << " | ";
        }std::cout << std::endl;
    }

void watershed(cv::Mat _src, cv::Mat _markers)
{
    // Labels for pixels
    const int IN_QUEUE = -2; // Pixel visited
    const int WSHED = -1; // Pixel belongs to watershed

    // possible bit values = 2^8
    const int NQ = 256;

    Mat src = _src, dst = _markers;
    Size size = src.size();

    // Vector of every created node
    std::vector<WSNode> storage;
    int free_node = 0, node;
    // Priority queue of queues of nodes
    // from high priority (0) to low priority (255)
    WSQueue q[NQ];
    // Non-empty queue with highest priority
    int active_queue;
    int i, j;
    // Color differences
    int db, dg, dr;
    int subs_tab[513];

    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
// MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])

// Create a new node with offsets mofs and iofs in queue idx
//#define ws_push(idx,mofs,iofs)          \
//    {                                       \
//        if( !free_node )                    \
//            free_node = allocWSNodes( storage );\
//        node = free_node;                   \
//        free_node = storage[free_node].next;\
//        storage[node].next = 0;             \
//        storage[node].mask_ofs = mofs;      \
//        storage[node].img_ofs = iofs;       \
//        if( q[idx].last )                   \
//            storage[q[idx].last].next=node; \
//        else                                \
//            q[idx].first = node;            \
//        q[idx].last = node;                 \
//    }

    // Get next node from queue idx
//#define ws_pop(idx,mofs,iofs)           \
//    {                                       \
//        node = q[idx].first;                \
//        q[idx].first = storage[node].next;  \
//        if( !storage[node].next )           \
//            q[idx].last = 0;                \
//        storage[node].next = free_node;     \
//        free_node = node;                   \
//        mofs = storage[node].mask_ofs;      \
//        iofs = storage[node].img_ofs;       \
//    }

        auto ws_push = [&](int idx, int mofs, int iofs)
        {
            std::cout << "push start==============" << std::endl;
            if (!free_node)
                free_node = allocWSNodes(storage);
            std::cout << "Freenode: " << free_node << std::endl;
            node = free_node;
            free_node = storage[free_node].next;
            storage[node].next = 0;
            storage[node].mask_ofs = mofs;
            storage[node].img_ofs = iofs;
            if (q[idx].last)
                storage[q[idx].last].next = node;
            else
                q[idx].first = node;
            q[idx].last = node;
            std::cout << "first last: " << q[idx].first << " " << q[idx].last << std::endl;
            std::cout << "push mofs: " << mofs / src.cols << " " << mofs % src.cols << " idx: "<<idx<<"push node: "<<node<<" freenode: "<< free_node <<std::endl;
            printBuf(dst);
            printVec(storage);
            std::cout << "push end+++++++++++++++++" << std::endl;
        };

        auto ws_pop = [&](int idx, int& mofs, int& iofs)
        {
            std::cout << "pop start--------------------" << std::endl;
            node = q[idx].first;
            q[idx].first = storage[node].next;
            if (!storage[node].next)
                q[idx].last = 0;
            storage[node].next = free_node;
            std::cout << "next: " << storage[node].next << " node:  "<< node<<std::endl;
            free_node = node;
            mofs = storage[node].mask_ofs;
            iofs = storage[node].img_ofs;
            printVec(storage);
            std::cout << "pop end++++++++++++++++++++++!!! freenode: " <<free_node<< std::endl;
        };

    // Get highest absolute channel difference in diff
#define c_diff(ptr1,ptr2,diff)           \
    {                                        \
        db = std::abs((ptr1)[0] - (ptr2)[0]);\
        dg = std::abs((ptr1)[1] - (ptr2)[1]);\
        dr = std::abs((ptr1)[2] - (ptr2)[2]);\
        diff = ws_max(db,dg);                \
        diff = ws_max(diff,dr);              \
        CV_Assert( 0 <= diff && diff <= 255 );  \
    }

    CV_Assert(src.type() == CV_8UC3 && dst.type() == CV_32SC1);
    CV_Assert(src.size() == dst.size());

    // Current pixel in input image
    const uchar* img = src.ptr();
    // Step size to next row in input image
    int istep = int(src.step / sizeof(img[0]));

    // Current pixel in mask image
    int* mask = dst.ptr<int>();
    // Step size to next row in mask image
    int mstep = int(dst.step / sizeof(mask[0]));

    for (i = 0; i < 256; i++)
        subs_tab[i] = 0;
    for (i = 256; i <= 512; i++)
        subs_tab[i] = i - 256;

    // draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for (j = 0; j < size.width; j++)
        mask[j] = mask[j + mstep * (size.height - 1)] = WSHED;

    // initial phase: put all the neighbor pixels of each marker to the ordered queue -
    // determine the initial boundaries of the basins
    for (i = 1; i < size.height - 1; i++)
    {
        img += istep; mask += mstep;
        mask[0] = mask[size.width - 1] = WSHED; // boundary pixels

        for (j = 1; j < size.width - 1; j++)
        {
            int* m = mask + j;
            if (m[0] < 0) m[0] = 0;
            if (m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0))
            {
                // Find smallest difference to adjacent markers
                const uchar* ptr = img + j * 3;
                int idx = 256, t;
                if (m[-1] > 0)
                    c_diff(ptr, ptr - 3, idx);
                if (m[1] > 0)
                {
                    c_diff(ptr, ptr + 3, t);
                    idx = ws_min(idx, t);
                }
                if (m[-mstep] > 0)
                {
                    c_diff(ptr, ptr - istep, t);
                    idx = ws_min(idx, t);
                }
                if (m[mstep] > 0)
                {
                    c_diff(ptr, ptr + istep, t);
                    idx = ws_min(idx, t);
                }

                // Add to according queue
                CV_Assert(0 <= idx && idx <= 255);
                ws_push(idx, i * mstep + j, i * istep + j * 3);
                m[0] = IN_QUEUE;
            }
        }
    }

    // find the first non-empty queue
    for (i = 0; i < NQ; i++)
        if (q[i].first)
            break;

    // if there is no markers, exit immediately
    if (i == NQ)
        return;

    active_queue = i;
    img = src.ptr();
    mask = dst.ptr<int>();

    // recursively fill the basins
    for (;;)
    {
        int mofs{}, iofs{};
        int lab = 0, t;
        int* m;
        const uchar* ptr;

        // Get non-empty queue with highest priority
        // Exit condition: empty priority queue
        if (q[active_queue].first == 0)
        {
            for (i = active_queue + 1; i < NQ; i++)
                if (q[i].first)
                    break;
            if (i == NQ)
                break;
            active_queue = i;
        }

        // Get next node
        ws_pop(active_queue, mofs, iofs);

        // Calculate pointer to current pixel in input and marker image
        m = mask + mofs;
        ptr = img + iofs;

        // Check surrounding pixels for labels
        // to determine label for current pixel
        t = m[-1]; // Left
        if (t > 0) lab = t;
        t = m[1]; // Right
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[-mstep]; // Top
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }
        t = m[mstep]; // Bottom
        if (t > 0)
        {
            if (lab == 0) lab = t;
            else if (t != lab) lab = WSHED;
        }

        // Set label to current pixel in marker image
        CV_Assert(lab != 0);
        m[0] = lab;
        std::cout<<"label location: "<< mofs / src.cols << " " << mofs % src.cols  << std::endl;

        if (lab == WSHED)
            continue;

        // Add adjacent, unlabeled pixels to corresponding queue
        if (m[-1] == 0)
        {
            c_diff(ptr, ptr - 3, t);
            ws_push(t, mofs - 1, iofs - 3);
            active_queue = ws_min(active_queue, t);
            m[-1] = IN_QUEUE;
        }
        if (m[1] == 0)
        {
            c_diff(ptr, ptr + 3, t);
            ws_push(t, mofs + 1, iofs + 3);
            active_queue = ws_min(active_queue, t);
            m[1] = IN_QUEUE;
        }
        if (m[-mstep] == 0)
        {
            c_diff(ptr, ptr - istep, t);
            ws_push(t, mofs - mstep, iofs - istep);
            active_queue = ws_min(active_queue, t);
            m[-mstep] = IN_QUEUE;
        }
        if (m[mstep] == 0)
        {
            c_diff(ptr, ptr + istep, t);
            ws_push(t, mofs + mstep, iofs + istep);
            active_queue = ws_min(active_queue, t);
            m[mstep] = IN_QUEUE;
        }

        printBuf(dst);
        std::cout << std::endl;
    }
}
}

void main(){
	cv::Mat src(30,30, CV_8UC3, cv::Scalar(0));
	cv::Mat mask(30, 30, CV_32SC1, Scalar(0));
	mask(Rect(5, 5, 3, 3)) = 1;
	mask(Rect(15, 5, 3, 3)) = 2;
	src(Rect(3, 3, 20, 20))=1;
	src.at<Vec3b>(4, 16) = { 5,5,5 };

	MWS::watershed(src, mask);
	cout << mask << endl;
}
