#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <climits>
#include <numeric>

// Copy necessary functions from analyser.cpp
struct PuzzleMove {
    std::string name;
    std::vector<int> perm;
    int group=-1;
};

std::vector<int> applyMove(const std::vector<int>& state,const PuzzleMove& m){
    std::vector<int> out(state.size());
    for(size_t i=0;i<state.size();++i) out[i]=state[m.perm[i]];
    return out;
}

std::string stateKey(const std::vector<int>& s){
    std::string k; k.reserve(s.size()*3);
    for(int v:s){k.append(std::to_string(v));k.push_back(',');}
    return k;
}

bool sameMultiset(const std::vector<int>&a,const std::vector<int>&b){
    if(a.size()!=b.size()) return false;
    std::unordered_map<int,int> c;
    for(int v:a) ++c[v];
    for(int v:b) if(--c[v]<0) return false;
    for(auto&q:c) if(q.second) return false;
    return true;
}

int countMisplaced(const std::vector<int>&s,const std::vector<int>&g){int c=0;for(size_t i=0;i<s.size();++i)if(s[i]!=g[i])++c;return c;}

struct IDAStarSolver {
    std::vector<PuzzleMove> moves;
    std::vector<int> goal,path;
    std::unordered_map<std::string,int> bestDepth;
    int maxAffected=1,nodes=0,maxNodes=2000000;
    bool solved=false;
    int heuristic(const std::vector<int>&s)const{return (countMisplaced(s,goal)+maxAffected-1)/maxAffected;}
    int dfs(const std::vector<int>&s,int g,int bound,int prevGroup){
        ++nodes;if(nodes>maxNodes)return INT_MAX/4;
        int h=heuristic(s),f=g+h;if(f>bound)return f;if(s==goal){solved=true;return g;}
        std::string key=stateKey(s);auto it=bestDepth.find(key);if(it!=bestDepth.end()&&it->second<=g)return INT_MAX/4;bestDepth[key]=g;
        int nextBound=INT_MAX/4;
        std::vector<int>order(moves.size());std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return countMisplaced(applyMove(s,moves[a]),goal)<countMisplaced(applyMove(s,moves[b]),goal);});
        for(int mi:order){if(moves[mi].group==prevGroup)continue;std::vector<int>ns=applyMove(s,moves[mi]);path.push_back(mi);int r=dfs(ns,g+1,bound,moves[mi].group);if(solved)return r;path.pop_back();if(r<nextBound)nextBound=r;if(nodes>maxNodes)return INT_MAX/4;}
        return nextBound;
    }
    std::vector<int> solve(const std::vector<int>&start,const std::vector<int>&g,int maxDepth=10,int nodeLimit=2000000){
        if(start.size()!=g.size()||!sameMultiset(start,g)) return{};
        goal=g; maxNodes=nodeLimit; nodes=0; solved=false; path.clear(); maxAffected=1;
        for(auto&m:moves){int a=0;for(size_t i=0;i<m.perm.size();++i)if(m.perm[i]!=(int)i)++a;maxAffected=std::max(maxAffected,a);}
        bestDepth.clear();
        int h=heuristic(start);for(int bound=h;bound<=maxDepth;++bound){bestDepth.clear();int r=dfs(start,0,bound,-1);if(solved)return path;if(r==INT_MAX/4)break;}return{};
    }
};

struct CubeSticker { int x=0,y=0,z=0; int nx=0,ny=0,nz=0; };
std::string stickerKey(const CubeSticker&s){return std::to_string(s.x)+":"+std::to_string(s.y)+":"+std::to_string(s.z)+":"+std::to_string(s.nx)+":"+std::to_string(s.ny)+":"+std::to_string(s.nz);}
std::vector<CubeSticker> makeCubeStickers(int N){
    std::vector<CubeSticker>v;int s=N-1;
    auto addFace=[&](int face){for(int r=0;r<N;++r)for(int c=0;c<N;++c){int x=0,y=0,z=0,nx=0,ny=0,nz=0;
        if(face==0){x=2*c-s;y=s;z=s-2*r;ny=1;}
        if(face==1){x=s;y=s-2*r;z=2*c-s;nx=1;}
        if(face==2){x=2*c-s;y=s-2*r;z=s;nz=1;}
        if(face==3){x=2*c-s;y=-s;z=2*r-s;ny=-1;}
        if(face==4){x=-s;y=s-2*r;z=s-2*c;nx=-1;}
        if(face==5){x=s-2*c;y=s-2*r;z=-s;nz=-1;}
        v.push_back({x,y,z,nx,ny,nz});}};
    for(int f=0;f<6;++f) addFace(f);
    return v;
}
CubeSticker rotateSticker(CubeSticker q,int axis,int signedAngle){
    auto rot=[&](int a,int b,int&oa,int&ob){if(signedAngle>0){oa=-b;ob=a;}else{oa=b;ob=-a;}};
    if(axis==0){int oa,ob;rot(q.y,q.z,oa,ob);q.y=oa;q.z=ob;rot(q.ny,q.nz,oa,ob);q.ny=oa;q.nz=ob;}
    if(axis==1){int oa,ob;rot(q.z,q.x,oa,ob);q.z=oa;q.x=ob;rot(q.nz,q.nx,oa,ob);q.nz=oa;q.nx=ob;}
    if(axis==2){int oa,ob;rot(q.x,q.y,oa,ob);q.x=oa;q.y=ob;rot(q.nx,q.ny,oa,ob);q.nx=oa;q.ny=ob;}
    return q;
}
std::vector<int> cubeFacePermutation(int N,int face,int turn){
    auto stickers=makeCubeStickers(N);std::unordered_map<std::string,int>index;for(int i=0;i<(int)stickers.size();++i)index[stickerKey(stickers[i])]=i;
    const int s=N-1;int axis=(face==1||face==4)?0:(face==0||face==3)?1:2;int sign=(face==0||face==1||face==2)?1:-1;int angle=turn*sign*-1;
    std::vector<int>perm(stickers.size());std::iota(perm.begin(),perm.end(),0);
    for(int i=0;i<(int)stickers.size();++i){auto q=stickers[i];int layer=(axis==0?q.x:(axis==1?q.y:q.z));int target=(axis==0?q.nx:(axis==1?q.ny:q.nz))*s;if(layer==target){auto r=rotateSticker(q,axis,angle);auto it=index.find(stickerKey(r));if(it==index.end())return{};perm[it->second]=i;}}
    return perm;
}
std::vector<PuzzleMove> makeCubeMoves(int N){
    std::vector<PuzzleMove>m;const std::string faces="URFDLB";for(int f=0;f<6;++f){auto cw=cubeFacePermutation(N,f,1);if(cw.empty())return{};auto ccw=cubeFacePermutation(N,f,-1);if(ccw.empty())return{};std::vector<int>dbl=cw;dbl= [&](){std::vector<int>v(dbl.size());for(size_t i=0;i<dbl.size();++i)v[i]=dbl[cw[i]];return v;}();m.push_back({std::string(1,faces[f]),cw,f});m.push_back({std::string(1,faces[f])+"'",ccw,f});m.push_back({std::string(1,faces[f])+"2",dbl,f});}return m;
}
std::vector<std::string> tokenizeMoves(const std::string&s){std::vector<std::string>v;std::istringstream in(s);std::string q;while(in>>q){if(q.size()==2&&(q[1]=='\''))v.push_back(q);else if(q.size()>=2&&q[1]=='2')v.push_back(q.substr(0,1)+"2");else v.push_back(q.substr(0,1));}return v;}

int main(){
    int N=3;
    auto moves=makeCubeMoves(N);
    std::vector<int> solved(6*N*N);for(int f=0;f<6;++f)for(int i=0;i<N*N;++i)solved[f*N*N+i]=f;
    
    // Print solved state
    std::cout << "SOLVED: ";
    for(int v: solved) std::cout << v; std::cout << "\n";
    
    // Apply B
    int b_idx = 5; // B move
    auto state = applyMove(solved, moves[b_idx*3]); // B (first of the 3: cw, ccw, double)
    std::cout << "AFTER_B: ";
    for(int v: state) std::cout << v; std::cout << "\n";
    
    // Apply U
    int u_idx = 0;
    state = applyMove(state, moves[u_idx*3]);
    std::cout << "AFTER_B_U: ";
    for(int v: state) std::cout << v; std::cout << "\n";
    
    state = applyMove(state, moves[u_idx*3]);
    std::cout << "AFTER_B_U2: ";
    for(int v: state) std::cout << v; std::cout << "\n";
    
    // Test the full scramble
    std::vector<int> scramble_state = solved;
    std::string scramble = "B U2 L R F2 U2 B U D U2 D2 B D2 L R2";
    std::istringstream iss(scramble);
    std::string tok;
    while(iss >> tok){
        int mi = -1;
        for(int i=0;i<(int)moves.size();++i){
            if(moves[i].name == tok){
                mi = i; break;
            }
        }
        if(mi>=0) scramble_state = applyMove(scramble_state, moves[mi]);
    }
    std::cout << "SCRAMBLE: ";
    for(int v: scramble_state) std::cout << v; std::cout << "\n";
    
    // Print move permutations for reference
    for(const auto& m : moves){
        std::cout << "MOVE " << m.name << ": ";
        for(size_t i=0;i<m.perm.size();++i){
            if(i) std::cout << ",";
            std::cout << m.perm[i];
        }
        std::cout << "\n";
    }
    
    return 0;
}