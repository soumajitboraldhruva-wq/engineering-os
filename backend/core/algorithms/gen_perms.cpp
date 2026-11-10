// gen_perms.cpp - Generate move permutations from the C++ engine
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

// Copy the necessary functions from analyser.cpp
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

std::vector<PuzzleMove> makeCubeMoves(int N){
    // [Copy from analyser.cpp lines 4896-4933]
    // ... (include the full makeCubeMoves implementation)
    // For brevity, we'll compile this with the actual analyser.cpp
    return {};
}

int main() {
    int N = 3;
    auto moves = makeCubeMoves(N);
    std::vector<int> solved(6*N*N);
    for(int f=0;f<6;++f)for(int i=0;i<N*N;++i)solved[f*N*N+i]=f;
    
    for(const auto& m : moves){
        std::vector<int> result = applyMove(solved, m);
        std::cout << m.name << ": ";
        for(size_t i=0;i<result.size();++i){
            if(i) std::cout << ",";
            std::cout << result[i];
        }
        std::cout << "\n";
    }
    return 0;
}