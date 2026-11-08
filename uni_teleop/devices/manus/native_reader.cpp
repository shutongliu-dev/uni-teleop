// Isolated MANUS C++ SDK reader. Uses public vendor headers; no robot commands.
#include <ManusSDK.h>
#include <ManusSDKTypeInitializers.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/prctl.h>
#include <time.h>

static std::atomic_bool running{true}, calibrated{false}, seen{false};
static uint32_t glove;
static std::mutex mutex;
static std::string latest;
static uint64_t sequence=0;
static const char* chain(ChainType c) {
 switch(c) {case ChainType_Hand:return "Hand";case ChainType_FingerThumb:return "FingerThumb";
 case ChainType_FingerIndex:return "FingerIndex";case ChainType_FingerMiddle:return "FingerMiddle";
 case ChainType_FingerRing:return "FingerRing";case ChainType_FingerPinky:return "FingerPinky";default:return "Unknown";}
}
static const char* joint(FingerJointType j) {
 switch(j) {case FingerJointType_Invalid:return "Invalid";case FingerJointType_Metacarpal:return "Metacarpal";
 case FingerJointType_Proximal:return "Proximal";case FingerJointType_Intermediate:return "Intermediate";
 case FingerJointType_Distal:return "Distal";case FingerJointType_Tip:return "Tip";default:return "Unknown";}
}
static void raw(const SkeletonStreamInfo* stream) {
 if(!stream)return;
 timespec ts{};clock_gettime(CLOCK_MONOTONIC,&ts);
 auto stamp=uint64_t(ts.tv_sec)*1000000000ULL+ts.tv_nsec;
 for(uint32_t i=0;i<stream->skeletonsCount;++i) {
  RawSkeletonInfo info{};RawSkeletonInfo_Init(&info);
  if(CoreSdk_GetRawSkeletonInfo(i,&info)!=SDKReturnCode_Success || info.nodesCount!=25)continue;
  std::vector<SkeletonNode> nodes(info.nodesCount);std::vector<NodeInfo> meta(info.nodesCount);
  if(CoreSdk_GetRawSkeletonData(i,nodes.data(),info.nodesCount)!=SDKReturnCode_Success ||
     CoreSdk_GetRawSkeletonNodeInfoArray(info.gloveId,meta.data(),info.nodesCount)!=SDKReturnCode_Success)continue;
  static std::set<uint32_t> reported;
  if(reported.insert(info.gloveId).second)fprintf(stderr,"Glove discovered: id=%u side=%s\n",info.gloveId,meta[0].side==Side_Right?"right":"left");
  if(info.gloveId!=glove || meta[0].side!=Side_Right)continue;
  seen=true;
  if(!calibrated)continue;
  std::set<uint32_t> ids;for(auto& m:meta)ids.insert(m.nodeId);
  std::ostringstream out;out<<std::setprecision(9)<<"{\"sequence\":"<<sequence++<<",\"capture_time_ns\":"<<stamp
      <<",\"source_timestamp\":\""<<stream->publishTime.time<<"\",\"glove_id\":"<<glove<<",\"nodes\":[";
  bool first=true;
  for(auto& n:nodes) {
   const NodeInfo* m=nullptr;for(auto& candidate:meta)if(candidate.nodeId==n.id)m=&candidate;
   if(!m)return;
   if(!first)out<<',';first=false;
   out<<"{\"node_id\":\""<<n.id<<"\",\"parent_id\":";
   if(m->parentId!=n.id && ids.count(m->parentId))out<<'"'<<m->parentId<<'"';else out<<"null";
   auto& p=n.transform.position;auto& q=n.transform.rotation;
   out<<",\"semantic\":\""<<chain(m->chainType)<<'/'<<joint(m->fingerJointType)
      <<"\",\"pose\":{\"position\":["<<p.x<<','<<p.y<<','<<p.z<<"],\"quaternion_wxyz\":["
      <<q.w<<','<<q.x<<','<<q.y<<','<<q.z<<"]},\"valid\":true}";
  }
  out<<"]}\n";std::lock_guard<std::mutex> lock(mutex);latest=out.str();
 }
}
int main(int argc,char** argv) {
 if(argc!=3)return 2;
 auto parent=getppid();prctl(PR_SET_PDEATHSIG,SIGTERM);if(getppid()!=parent)return 2;
 glove=std::stoul(argv[1]);
 std::ifstream f(argv[2],std::ios::binary);std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)),{});
 if(data.empty())return 2;
 FILE* output=fdopen(dup(STDOUT_FILENO),"w");dup2(STDERR_FILENO,STDOUT_FILENO);
 std::signal(SIGTERM,[](int){running=false;});std::signal(SIGINT,[](int){running=false;});
 if(CoreSdk_InitializeIntegrated()!=SDKReturnCode_Success)return 2;
 struct Shutdown{~Shutdown(){CoreSdk_ShutDown();}} shutdown;
 if(CoreSdk_RegisterCallbackForRawSkeletonStream(raw)!=SDKReturnCode_Success)return 2;
 CoordinateSystemVUH coordinates{};CoordinateSystemVUH_Init(&coordinates);
 coordinates.handedness=Side_Right;coordinates.up=AxisPolarity_PositiveZ;coordinates.view=AxisView_XFromViewer;coordinates.unitScale=1.f;
 if(CoreSdk_InitializeCoordinateSystemWithVUH(coordinates,true)!=SDKReturnCode_Success)return 2;
 bool connected=false;auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(20);
 while(running && std::chrono::steady_clock::now()<deadline && !connected) {
  uint32_t count=0;
  if(CoreSdk_LookForHosts(1,false)==SDKReturnCode_Success && CoreSdk_GetNumberOfAvailableHostsFound(&count)==SDKReturnCode_Success && count==1) {
   ManusHost host{};if(CoreSdk_GetAvailableHostsFound(&host,1)==SDKReturnCode_Success)
    connected=CoreSdk_ConnectToHost(host)==SDKReturnCode_Success;
  }
 }
 if(!running || !connected)return 2;
 // Match the vendor finger-only stream; PICO owns wrist/arm tracking.
 if(CoreSdk_SetRawSkeletonHandMotion(HandMotion_None)!=SDKReturnCode_Success)return 2;
 deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
 while(running && !seen && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::milliseconds(20));
 if(!running || !seen)return 2;
 SetGloveCalibrationReturnCode result{};
 auto code=CoreSdk_SetGloveCalibration(glove,data.data(),data.size(),&result);
 if(code!=SDKReturnCode_Success || result!=SetGloveCalibrationReturnCode_Success) {
  fprintf(stderr,"Calibration load failed: sdk=%d calibration=%d\n",int(code),int(result));return 3;
 }
 fprintf(stderr,"Selected right glove calibration loaded successfully\n");calibrated=true;
 while(running) {
  std::string packet;{std::lock_guard<std::mutex> lock(mutex);packet.swap(latest);}
  if(!packet.empty()){fwrite(packet.data(),1,packet.size(),output);fflush(output);}
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
 }
 return 0;
}
