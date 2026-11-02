#include <algorithm>
#include <bitset>
#include <chrono>
#include <complex>
#include <cstring>
#include <filesystem>
#include <array>
#include <deque>
#include <map>
#include <unordered_map>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#ifdef near
#undef near
#endif
#include <bcrypt.h>
#endif

#ifdef ENGINE_USE_OPENSSL
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif

using namespace std;
using ld = long double;

constexpr ld EPS = 1e-12L;
constexpr ld PI = 3.141592653589793238462643383279502884L;

bool isUnitSpacedGrid(const vector<ld>& x);


struct Point { ld x{}, y{}, z{}; };

struct Stats {
    int n = 0;
    ld mean = 0, median = 0, variance = 0, sd = 0;
    ld minv = 0, maxv = 0, q1 = 0, q3 = 0, iqr = 0, mad = 0;
    ld skew = 0, excessKurtosis = 0;
    ld lag1 = numeric_limits<ld>::quiet_NaN();
    ld durbinWatson = numeric_limits<ld>::quiet_NaN();
    ld runsZ = numeric_limits<ld>::quiet_NaN();
    ld jarqueBera = numeric_limits<ld>::quiet_NaN();
    ld jarqueBeraP = numeric_limits<ld>::quiet_NaN();
    ld standardError = numeric_limits<ld>::quiet_NaN();
    ld ci95Low = numeric_limits<ld>::quiet_NaN();
    ld ci95High = numeric_limits<ld>::quiet_NaN();
};

struct Metrics {
    int nTrain = 0;
    int nValid = 0;
    int validationBlocks = 0;
    ld rmse = numeric_limits<ld>::infinity();
    ld mae = numeric_limits<ld>::infinity();
    ld r2 = -numeric_limits<ld>::infinity();
    ld normalizedRMSE = numeric_limits<ld>::infinity();
    ld hitRate = 0;
    ld meanDeviationPct = numeric_limits<ld>::infinity();
    ld maxDeviationPct = numeric_limits<ld>::infinity();
    ld p90DeviationPct = numeric_limits<ld>::infinity();
    ld smapePct = numeric_limits<ld>::infinity();
    ld directionalAccuracy = numeric_limits<ld>::quiet_NaN();
    ld correlation = numeric_limits<ld>::quiet_NaN();
    ld wapePct = numeric_limits<ld>::infinity();
    ld mase = numeric_limits<ld>::infinity();
    ld medianDeviationPct = numeric_limits<ld>::infinity();
    ld blockPassRate = 0;
    ld score = 0;
    ld scoreRMSE = 0, scoreMAE = 0, scoreP90 = 0, scoreR2 = 0, scoreDirection = 0, scoreCorrelation = 0, scoreStability = 0, scoreComplexity = 0;
    bool stable = false;
    bool rollingValidated = false;
    vector<ld> blockMeanDeviationPct;
    vector<ld> blockMaxDeviationPct;
};

struct Model {
    string name;
    string category;
    string formula;
    int complexity = 1;
    bool valid = false;
    bool exact = false;
    function<ld(ld)> predict;
    vector<ld> fitted;
    vector<ld> validationResiduals;
    Metrics metrics;
    ld nextValue = numeric_limits<ld>::quiet_NaN();
    int window = 0;
    string note;
    vector<ld> approxRealRoots;
    vector<ld> approxDerivativeRoots;
    vector<ld> approxInflectionRoots;
    string calculusSummary;
};

struct ForecastOptions {
    int horizon = 1;
    bool customTolerance = false;
    ld tolerancePct = 0;
    bool requestPlot = false;
};

struct Analysis {
    bool found = false;
    bool deterministicNext = false;
    string category;
    string logic;
    string formula;
    ld nextValue = numeric_limits<ld>::quiet_NaN();
    ld evidenceScore = 0;
    ld tolerancePct = 0;
    Model model;
    vector<Model> leaderboard;
    vector<ld> predictions;
    vector<ld> predictionLow;
    vector<ld> predictionHigh;
    vector<int> changePoints;
    vector<ld> observedZeroCrossings;
    vector<ld> observedTurningPoints;
    vector<ld> observedInflectionPoints;
    string calculusSummary;
    bool shapeBased = false;
    string warning;
};

struct RelationAnalysis {
    bool found = false;
    string category;
    string logic;
    string formula;
    ld evidenceScore = 0;
    ld tolerancePct = 0;
    Model model;
    vector<Model> leaderboard;
    vector<ld> futurePredictions;
    string warning;
};

struct MultiAnalysis {
    bool validStructure = false;
    bool fullForecastAvailable = false;
    int variables = 0;
    int rows = 0;
    ld tolerancePct = 0;
    Analysis driver;
    vector<RelationAnalysis> links;
    vector<vector<ld>> predictions;
    string failureReason;
};

// ----------------------------- Formatting -----------------------
string fmt(ld x, int precision = 8) {
    if (!isfinite(x)) return isnan(x) ? "NaN" : (x > 0 ? "+Inf" : "-Inf");
    if (fabsl(x) < 0.5e-10L) x = 0;
    if (fabsl(x - roundl(x)) < 1e-9L && fabsl(x) < 9.22e18L)
        return to_string((long long)llroundl(x));
    ostringstream ss;
    ss << scientific << setprecision(precision) << x;
    return ss.str();
}

string fmtFixed(ld x, int decimals = 3) {
    if (!isfinite(x)) return isnan(x) ? "NaN" : (x > 0 ? "+Inf" : "-Inf");
    ostringstream ss;
    ss << fixed << setprecision(decimals) << (double)x;
    return ss.str();
}

int sizeToInt(size_t value) {
    if (value > static_cast<size_t>(INT_MAX)) return INT_MAX;
    return static_cast<int>(value);
}

string pyQuote(const string& s) {
    string out = "\"";
    for (char c : s) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else out += c;
    }
    out += '"';
    return out;
}

bool approxEqual(ld a, ld b, ld rel = 1e-9L, ld absTol = 1e-9L) {
    return fabsl(a - b) <= absTol + rel * max({fabsl(a), fabsl(b), 1.0L});
}

bool isInteger(ld x) {
    return isfinite(x) && fabsl(x - roundl(x)) < 1e-9L;
}

ld clamp01(ld x) { return max<ld>(0, min<ld>(1, x)); }
ld clamp(ld x, ld lo, ld hi) { return max(lo, min(hi, x)); }

// ----------------------------- Numeric helpers ------------------
ld meanOf(const vector<ld>& v) {
    if (v.empty()) return 0;
    return accumulate(v.begin(), v.end(), 0.0L) / (ld)v.size();
}

ld varianceOf(const vector<ld>& v, bool sample = true) {
    if (v.size() < (sample ? 2u : 1u)) return 0;
    ld m = meanOf(v), s = 0;
    for (ld x : v) s += (x - m) * (x - m);
    return s / (ld)(v.size() - (sample ? 1 : 0));
}

ld quantile(vector<ld> v, ld q) {
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    q = clamp01(q);
    ld pos = q * (v.size() - 1);
    size_t lo = (size_t)floorl(pos), hi = (size_t)ceill(pos);
    if (lo == hi) return v[lo];
    ld t = pos - lo;
    return v[lo] * (1 - t) + v[hi] * t;
}

ld robustScale(const vector<ld>& y) {
    if (y.empty()) return 1;
    ld sd = sqrtl(max<ld>(0, varianceOf(y)));
    ld q1 = quantile(y, .25L), q3 = quantile(y, .75L);
    ld iqr = q3 - q1;
    return max({sd, iqr / 1.349L, 1e-12L});
}

ld r2Score(const vector<ld>& y, const vector<ld>& p) {
    if (y.empty() || y.size() != p.size()) return -numeric_limits<ld>::infinity();
    ld m = meanOf(y), st = 0, sr = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        st += (y[i] - m) * (y[i] - m);
        sr += (y[i] - p[i]) * (y[i] - p[i]);
    }
    if (st < 1e-24L) return sr < 1e-24L ? 1 : 0;
    return 1 - sr / st;
}

pair<ld, ld> errorMetrics(const vector<ld>& y, const vector<ld>& p) {
    if (y.empty() || y.size() != p.size())
        return {numeric_limits<ld>::infinity(), numeric_limits<ld>::infinity()};
    ld sse = 0, sae = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        ld e = y[i] - p[i];
        sse += e * e;
        sae += fabsl(e);
    }
    return {sqrtl(sse / y.size()), sae / y.size()};
}

ld medianAbs(const vector<ld>& y) {
    vector<ld> a;
    a.reserve(y.size());
    for (ld v : y) a.push_back(fabsl(v));
    return quantile(a, .5L);
}

// Data-driven one-step noise scale. It is deliberately a descriptive threshold,
// not a confidence level. The threshold is the 90th percentile of robust
// first-difference noise relative to a robust level scale.
ld adaptiveTolerancePct(const vector<ld>& y) {
    if (y.size() < 5) return 5.0L;
    vector<ld> localResiduals;
    localResiduals.reserve(y.size()-2);
    for(size_t i=1;i+1<y.size();++i){
        ld localLinear=0.5L*(y[i-1]+y[i+1]);
        localResiduals.push_back(fabsl(y[i]-localLinear));
    }
    ld q90=quantile(localResiduals,.90L);
    ld level=max({medianAbs(y),1.4826L*quantile([&](){vector<ld>d;d.reserve(y.size()-1);for(size_t i=1;i<y.size();++i)d.push_back(y[i]-y[i-1]);return d;}(),.75L),1e-12L});
    // The numerator measures local departure from linear interpolation rather
    // than raw first differences, so smooth curvature is not mistaken for noise.
    return clamp(100.0L*q90/max<ld>(level,1e-12L),1.0L,20.0L);
}

ld relativeDeviationPct(ld actual, ld predicted, ld scale) {
    const ld denom = max({fabsl(actual), scale, 1e-12L});
    return 100.0L * fabsl(predicted - actual) / denom;
}


vector<ld> finiteFitted(const Model& m, const vector<ld>& x, const vector<ld>& fallbackY) {
    vector<ld> p;
    p.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        ld v = m.predict ? m.predict(x[i]) : numeric_limits<ld>::quiet_NaN();
        p.push_back(isfinite(v) ? v : fallbackY[i]);
    }
    return p;
}

vector<ld> forecastFromModel(const Model& m, ld lastX, int horizon) {
    vector<ld> out;
    if (!m.predict || horizon <= 0) return out;
    out.reserve(horizon);
    for (int h = 1; h <= horizon; ++h) {
        ld v = m.predict(lastX + (ld)h);
        if (!isfinite(v)) return {};
        out.push_back(v);
    }
    return out;
}

// ----------------------------- Linear algebra -------------------
bool leastSquaresQR(const vector<vector<ld>>& Ain, const vector<ld>& bin, vector<ld>& x) {
    if (Ain.empty() || Ain.size() != bin.size()) return false;
    const auto m = sizeToInt(Ain.size());
    const auto p = sizeToInt(Ain[0].size());
    if (p == 0 || m < p) return false;
    for (const auto& row : Ain) if ((int)row.size() != p) return false;

    vector<vector<ld>> A = Ain;
    vector<ld> b = bin;
    for (int k = 0; k < p; ++k) {
        ld norm = 0;
        for (int i = k; i < m; ++i) norm = hypotl(norm, A[i][k]);
        if (norm < 1e-18L) return false;
        ld alpha = (A[k][k] >= 0 ? -norm : norm);
        vector<ld> v(m - k, 0);
        for (int i = k; i < m; ++i) v[i - k] = A[i][k];
        v[0] -= alpha;
        ld vv = inner_product(v.begin(), v.end(), v.begin(), 0.0L);
        if (vv < 1e-24L) return false;
        for (int j = k; j < p; ++j) {
            ld dot = 0;
            for (int i = k; i < m; ++i) dot += v[i - k] * A[i][j];
            ld tau = 2 * dot / vv;
            for (int i = k; i < m; ++i) A[i][j] -= tau * v[i - k];
        }
        ld dotb = 0;
        for (int i = k; i < m; ++i) dotb += v[i - k] * b[i];
        ld taub = 2 * dotb / vv;
        for (int i = k; i < m; ++i) b[i] -= taub * v[i - k];
    }
    x.assign(p, 0);
    for (int i = p - 1; i >= 0; --i) {
        ld s = b[i];
        for (int j = i + 1; j < p; ++j) s -= A[i][j] * x[j];
        if (fabsl(A[i][i]) < 1e-18L) return false;
        x[i] = s / A[i][i];
    }
    return all_of(x.begin(), x.end(), [](ld v) { return isfinite(v); });
}

vector<ld> linearFit(const vector<ld>& x, const vector<ld>& y) {
    vector<vector<ld>> A;
    A.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) A.push_back({1.0L, x[i]});
    vector<ld> c;
    if (!leastSquaresQR(A, y, c)) return {};
    return c;
}

// Dense square solve (Gaussian elimination, partial pivoting). Used by
// kernel/RBF regression, whose normal equations are square (n x n) rather
// than the tall rectangular systems leastSquaresQR is built for.
bool solveSquareSystem(vector<vector<ld>> A, vector<ld> b, vector<ld>& x) {
    const auto n = sizeToInt(A.size());
    if (n == 0 || (int)b.size() != n) return false;
    for (const auto& row : A) if ((int)row.size() != n) return false;
    for (int col = 0; col < n; ++col) {
        int piv = col; ld best = fabsl(A[col][col]);
        for (int r = col + 1; r < n; ++r) if (fabsl(A[r][col]) > best) { best = fabsl(A[r][col]); piv = r; }
        if (best < 1e-14L) return false;
        swap(A[col], A[piv]); swap(b[col], b[piv]);
        for (int r = col + 1; r < n; ++r) {
            ld f = A[r][col] / A[col][col];
            if (f == 0) continue;
            for (int c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    x.assign(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        ld s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        if (fabsl(A[i][i]) < 1e-14L) return false;
        x[i] = s / A[i][i];
    }
    return all_of(x.begin(), x.end(), [](ld v) { return isfinite(v); });
}

// Durand-Kerner (Weierstrass) simultaneous iteration for all roots of a
// real-coefficient polynomial, coefficients ascending (c[0] + c[1]t + ...).
// Returns the roots that are real to within a small relative tolerance,
// sorted ascending. Used to report approximate zero-crossings for
// polynomial-family fits, and to root-find the Chebyshev-basis model below.
vector<ld> polynomialRealRoots(const vector<ld>& coeffsAscending) {
    int n = (int)coeffsAscending.size() - 1;
    while (n > 0 && fabsl(coeffsAscending[n]) < 1e-14L) --n;
    if (n < 1 || n > 40) return {}; // guard against a degenerate or pathologically high degree
    vector<ld> c(coeffsAscending.begin(), coeffsAscending.begin() + n + 1);
    for (auto& v : c) v /= c[n]; // monic

    using cd = complex<ld>;
    vector<cd> roots(n);
    ld bound = 1;
    for (int i = 0; i < n; ++i) bound = max(bound, 1 + fabsl(c[i]));
    cd seed(0.4L, 0.9L), cur(1.0L, 0.0L);
    for (int i = 0; i < n; ++i) { roots[i] = cur * (bound * 0.5L); cur *= seed; }

    auto evalP = [&](cd z) { cd s(0, 0), p(1, 0); for (int i = 0; i <= n; ++i) { s += c[i] * p; p *= z; } return s; };
    for (int iter = 0; iter < 300; ++iter) {
        ld maxShift = 0;
        for (int i = 0; i < n; ++i) {
            cd denom(1, 0);
            for (int j = 0; j < n; ++j) if (j != i) denom *= (roots[i] - roots[j]);
            if (abs(denom) < 1e-30L) continue;
            cd shift = evalP(roots[i]) / denom;
            roots[i] -= shift;
            maxShift = max(maxShift, (ld)abs(shift));
        }
        if (maxShift < 1e-15L) break;
    }
    vector<ld> realRoots;
    for (const auto& r : roots) {
        ld scale = max<ld>(1e-9L, abs(r));
        if (fabsl(r.imag()) < 1e-6L * scale + 1e-9L) realRoots.push_back(r.real());
    }
    sort(realRoots.begin(), realRoots.end());
    return realRoots;
}

ld bisectRoot(const function<ld(ld)>& f, ld a, ld b, ld fa, ld fb) {
    if (!isfinite(fa) || !isfinite(fb)) return numeric_limits<ld>::quiet_NaN();
    if (fabsl(fa) < 1e-14L) return a;
    if (fabsl(fb) < 1e-14L) return b;
    if ((fa > 0 && fb > 0) || (fa < 0 && fb < 0)) return numeric_limits<ld>::quiet_NaN();
    for (int it = 0; it < 100; ++it) {
        ld m = (a + b) * 0.5L, fm = f(m);
        if (!isfinite(fm)) return numeric_limits<ld>::quiet_NaN();
        if (fabsl(fm) < 1e-14L) return m;
        if (fabsl(b-a) < 1e-13L*max<ld>(1.0L,fabsl(m))) {
            const ld scale = max({1.0L, fabsl(fa), fabsl(fb)});
            return fabsl(fm) <= 1e-8L*scale ? m : numeric_limits<ld>::quiet_NaN();
        }
        if ((fa < 0 && fm > 0) || (fa > 0 && fm < 0)) { b=m; fb=fm; }
        else { a=m; fa=fm; }
    }
    return (a+b)*0.5L;
}

vector<ld> polyDerivativeCoeffs(const vector<ld>& c){
    if(c.size()<2)return {};
    vector<ld>d(c.size()-1);
    for(size_t i=1;i<c.size();++i)d[i-1]=(ld)i*c[i];
    return d;
}

vector<ld> polyShiftVariable(const vector<ld>& c, ld x0){
    // Return coefficients in x for P(x-x0), given coefficients in t.
    // Horner composition: q(x)=c[0]+t(c[1]+t(...)), t=x-x0.
    vector<ld>q(1,0);
    for(auto it=c.rbegin();it!=c.rend();++it){
        vector<ld>nq(q.size()+1,0);
        nq[0]=*it-x0*(q.empty()?0:q[0]);
        for(size_t j=1;j<q.size();++j)nq[j]=q[j-1]-x0*q[j];
        nq[q.size()]=q.back();
        q.swap(nq);
    }
    while(q.size()>1&&fabsl(q.back())<1e-18L)q.pop_back();
    return q;
}

vector<ld> numericRealRoots(const function<ld(ld)>& f, ld xmin, ld xmax, int samples=2048) {
    vector<ld> roots;
    if(!f || !isfinite(xmin) || !isfinite(xmax) || xmax<=xmin) return roots;
    samples=max(64,min(samples,12000));
    auto add=[&](ld r){
        if(!isfinite(r) || r<xmin-1e-10L || r>xmax+1e-10L) return;
        for(ld q:roots) if(fabsl(r-q)<=1e-7L*max<ld>(1.0L,fabsl(r))) return;
        roots.push_back(clamp(r,xmin,xmax));
    };
    vector<ld> gx(samples+1),gv(samples+1);
    ld maxAbs=0;
    for(int i=0;i<=samples;++i){gx[i]=xmin+(xmax-xmin)*(ld)i/samples;gv[i]=f(gx[i]);if(isfinite(gv[i]))maxAbs=max(maxAbs,fabsl(gv[i]));}
    const ld nearZero=max<ld>(1e-10L,1e-8L*max<ld>(1.0L,maxAbs));
    // A function that is essentially zero over a long interval has an interval
    // of roots (or, for a numerical derivative, a flat region), not thousands
    // of distinct isolated roots. Suppress those samples so diagnostics remain
    // meaningful for plateaus, saturation regions, and piecewise signals.
    vector<char>flat(samples+1,0);
    int run=0;
    for(int i=0;i<=samples;++i){
        if(isfinite(gv[i])&&fabsl(gv[i])<=nearZero){flat[i]=1;++run;}
        else run=0;
        if(run>=max(6,samples/256))for(int j=i-run+1;j<=i;++j)flat[j]=1;
    }
    for(int i=0;i<=samples;++i) if(!flat[i]&&isfinite(gv[i])&&fabsl(gv[i])<=nearZero) add(gx[i]);
    for(int i=1;i<=samples;++i){
        if(isfinite(gv[i-1])&&isfinite(gv[i])&&((gv[i-1]<0&&gv[i]>0)||(gv[i-1]>0&&gv[i]<0))){ld r=bisectRoot(f,gx[i-1],gx[i],gv[i-1],gv[i]);if(isfinite(r)){ld fr=f(r);if(isfinite(fr)&&fabsl(fr)<=nearZero*10.0L)add(r);}}
    }
    // Catch even-multiplicity/tangential roots where there is no sign change.
    for(int i=1;i<samples;++i){
        if(!isfinite(gv[i-1])||!isfinite(gv[i])||!isfinite(gv[i+1]))continue;
        ld a=fabsl(gv[i-1]),b=fabsl(gv[i]),c=fabsl(gv[i+1]);
        if(b>a||b>c||b>nearZero*1000.0L)continue;
        ld lo=gx[i-1],hi=gx[i+1];
        for(int it=0;it<60;++it){
            const ld m1=lo+(hi-lo)*0.382L,m2=lo+(hi-lo)*0.618L;
            if(fabsl(f(m1))<fabsl(f(m2)))hi=m2;else lo=m1;
        }
        ld r=(lo+hi)*0.5L,v=f(r);
        int ri=(int)llround((ld)samples*(r-xmin)/max<ld>(xmax-xmin,1e-18L));
        if(ri>=0&&ri<=samples&&flat[ri])continue;
        if(isfinite(v)&&fabsl(v)<=nearZero*10.0L)add(r);
    }
    sort(roots.begin(),roots.end());
    return roots;
}

vector<ld> numericDerivativeRoots(const function<ld(ld)>& f, ld xmin, ld xmax, int samples=1024) {
    if(!f || xmax<=xmin) return {};
    auto df=[f,xmin,xmax](ld z){
        ld h=max<ld>(1e-6L,1e-5L*max({1.0L,fabsl(z),xmax-xmin}));
        h=min(h,max<ld>((xmax-xmin)*0.01L,1e-8L));
        ld a=f(z-h),b=f(z+h);
        return isfinite(a)&&isfinite(b)?(b-a)/(2*h):numeric_limits<ld>::quiet_NaN();
    };
    return numericRealRoots(df,xmin,xmax,samples);
}

void attachNumericRoots(Model& m,const vector<ld>& x,bool derivatives=true){
    if(!m.valid||!m.predict||x.size()<3)return;
    // For piecewise/signal models, derivative roots of a globally non-smooth
    // callable can be misleading.  The observed-data calculus diagnostics are
    // reported separately, while smooth local models still receive numerical
    // derivative/inflection roots here.
    const bool nonsmooth = m.name.find("piecewise")!=string::npos ||
                           m.name.find("Segmented")!=string::npos ||
                           m.name.find("zero-order")!=string::npos;
    const bool discrete = m.name.find("Prony")!=string::npos ||
                          m.name.find("autoregression")!=string::npos ||
                          m.name.find("AR(")!=string::npos ||
                          m.name.find("Fibonacci")!=string::npos ||
                          m.name.find("Arithmetic progression")!=string::npos ||
                          m.name.find("Geometric progression")!=string::npos ||
                          (m.name.find("Periodic")!=string::npos && m.name.find("regression")!=string::npos);
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end());
    if(m.name.find("Finite-difference polynomial")!=string::npos){
        // Coefficient-based roots/derivatives were computed when the model was
        // fit.  Keep them; do not replace them with coarse sampling.
        return;
    }
    if(m.name.find("piecewise-constant")!=string::npos || m.name.find("zero-order-hold")!=string::npos){
        m.approxRealRoots.clear();m.approxDerivativeRoots.clear();m.approxInflectionRoots.clear();return;
    }
    const int samples=(int)clamp<ld>((ld)x.size()*56.0L,512.0L,6000.0L);
    vector<ld> numericRoots;
    if(m.approxRealRoots.empty()) numericRoots=numericRealRoots(m.predict,xmin,xmax,samples);
    else numericRoots=m.approxRealRoots;
    if(!nonsmooth&&!discrete&&xmax>xmin){
        const ld span=max<ld>(xmax-xmin,1e-9L);
        auto extra=numericRealRoots(m.predict,xmin-2.0L*span,xmax+2.0L*span,min(7000,max(1024,samples)));
        for(ld r:extra){bool seen=false;for(ld q:numericRoots)if(fabsl(r-q)<=1e-7L*max<ld>(1.0L,fabsl(r))){seen=true;break;}if(!seen)numericRoots.push_back(r);}
        sort(numericRoots.begin(),numericRoots.end());
    }
    m.approxRealRoots=numericRoots;
    if(derivatives&&!nonsmooth&&!discrete){
        const ld dspan=max<ld>(xmax-xmin,1e-9L);
        m.approxDerivativeRoots=numericDerivativeRoots(m.predict,xmin,xmax+0.75L*dspan,max(256,samples/2));
        auto d2=[&](ld z){
            ld h=max<ld>(1e-5L,1e-4L*max({1.0L,fabsl(z),xmax-xmin}));
            h=min(h,max<ld>((xmax-xmin)*0.005L,1e-7L));
            ld a=m.predict(z-h),b=m.predict(z),c=m.predict(z+h);
            if(!isfinite(a)||!isfinite(b)||!isfinite(c))return numeric_limits<ld>::quiet_NaN();
            return (a-2*b+c)/(h*h);
        };
        m.approxInflectionRoots=numericRealRoots(d2,xmin,xmax+0.75L*dspan,max(256,samples/2));
        if(!m.approxDerivativeRoots.empty()||!m.approxInflectionRoots.empty()){
            ostringstream os;
            if(!m.approxDerivativeRoots.empty()){
                os<<"Numerical calculus: derivative roots (extrema candidates)=";
                for(size_t i=0;i<m.approxDerivativeRoots.size()&&i<12;++i){if(i)os<<",";os<<fmt(m.approxDerivativeRoots[i],5);}
            }
            if(!m.approxInflectionRoots.empty()){
                if(!m.approxDerivativeRoots.empty())os<<"; ";
                os<<"second-derivative roots (inflection candidates)=";
                for(size_t i=0;i<m.approxInflectionRoots.size()&&i<12;++i){if(i)os<<",";os<<fmt(m.approxInflectionRoots[i],5);}
            }
            m.calculusSummary=os.str();
        }
    }
}


vector<ld> observedZeroCrossings(const vector<ld>&x,const vector<ld>&y){
    vector<ld> roots;
    if(x.size()!=y.size()||x.size()<2)return roots;
    ld scale=max<ld>(robustScale(y),1e-12L),near=max<ld>(1e-10L,1e-7L*scale);
    for(size_t i=0;i<y.size();++i)if(fabsl(y[i])<=near)roots.push_back(x[i]);
    for(size_t i=1;i<y.size();++i){
        ld a=y[i-1],b=y[i];
        if(!isfinite(a)||!isfinite(b))continue;
        if((a<0&&b>0)||(a>0&&b<0)){
            ld t=a/(a-b),r=x[i-1]+t*(x[i]-x[i-1]);
            if(isfinite(r))roots.push_back(r);
        }
    }
    sort(roots.begin(),roots.end());
    vector<ld>u;for(ld r:roots)if(u.empty()||fabsl(r-u.back())>1e-7L*max<ld>(1.0L,fabsl(r)))u.push_back(r);
    return u;
}

vector<ld> observedTurningPoints(const vector<ld>&x,const vector<ld>&y){
    vector<ld>out;if(x.size()!=y.size()||x.size()<5)return out;
    vector<ld>d(x.size(),numeric_limits<ld>::quiet_NaN());
    // Five-point central derivative for smooth data.  Falling back to a central
    // difference keeps the routine usable on short/nonuniform domains.
    for(size_t i=2;i+2<x.size();++i){
        const ld h1=x[i]-x[i-1],h2=x[i+1]-x[i];
        if(fabsl(h1-h2)<1e-8L*max<ld>(1.0L,h1+h2)) d[i]=(y[i-2]-8*y[i-1]+8*y[i+1]-y[i+2])/(12*max<ld>(h1,1e-12L));
        else d[i]=(y[i+1]-y[i-1])/max<ld>(x[i+1]-x[i-1],1e-12L);
    }
    vector<pair<ld,int>> raw;
    for(size_t i=2;i+2<x.size();++i){
        ld a=d[i-1],b=d[i+1];if(!isfinite(a)||!isfinite(b))continue;
        if(a>0&&b<0)raw.push_back({x[i],+1});      // local maximum candidate
        else if(a<0&&b>0)raw.push_back({x[i],-1}); // local minimum candidate
    }
    if(raw.empty())return out;
    // Smooth the common two/three-sample duplicate detections around one extremum.
    vector<ld>dx;for(size_t i=1;i<x.size();++i){ld q=x[i]-x[i-1];if(q>0)dx.push_back(q);}ld mdx=dx.empty()?1.0L:quantile(dx,.5L);ld clusterGap=max<ld>(2.1L*mdx,1e-9L);
    for(int typ:{1,-1}){
        vector<ld>group;for(auto pr:raw)if(pr.second==typ)group.push_back(pr.first);if(group.empty())continue;
        sort(group.begin(),group.end());vector<vector<ld>>clusters;for(ld z:group){if(clusters.empty()||z-clusters.back().back()>clusterGap)clusters.push_back({z});else clusters.back().push_back(z);}
        for(const auto&cl:clusters){
            ld center=meanOf(cl);auto it=lower_bound(x.begin(),x.end(),center);size_t j=(size_t)(it-x.begin());if(j>=x.size())j=x.size()-1;
            size_t lo=(j>1?j-1:0),hi=min(x.size()-1,j+1),bestI=j;
            for(size_t q=lo;q<=hi;++q){if(typ>0?y[q]>y[bestI]:y[q]<y[bestI])bestI=q;}
            out.push_back(x[bestI]);
        }
    }
    sort(out.begin(),out.end());
    vector<ld>u;for(ld r:out)if(u.empty()||fabsl(r-u.back())>1e-7L*max<ld>(1.0L,fabsl(r)))u.push_back(r);
    return u;
}

vector<ld> observedInflectionPoints(const vector<ld>&x,const vector<ld>&y){
    vector<ld>out;if(x.size()!=y.size()||x.size()<4)return out;
    vector<ld>sl;
    for(size_t i=1;i<x.size();++i){ld dx=x[i]-x[i-1];sl.push_back(fabsl(dx)>1e-18L?(y[i]-y[i-1])/dx:numeric_limits<ld>::quiet_NaN());}
    vector<ld>curv;curv.reserve(sl.size()-1);
    for(size_t i=1;i<sl.size();++i){ld dx=0.5L*((x[i]-x[i-1])+(x[i+1]-x[i]));curv.push_back(fabsl(dx)>1e-18L?(sl[i]-sl[i-1])/dx:numeric_limits<ld>::quiet_NaN());}
    ld noise=max({medianAbs(curv),robustScale(curv)*0.05L,1e-10L});
    for(size_t i=1;i<curv.size();++i){ld a=curv[i-1],b=curv[i];if(!isfinite(a)||!isfinite(b))continue;if((a>noise&&b<-noise)||(a<-noise&&b>noise)){ld t=a/(a-b);ld r=x[i]+t*(x[i+1]-x[i]);if(isfinite(r))out.push_back(r);}}
    sort(out.begin(),out.end());vector<ld>u;for(ld r:out)if(u.empty()||fabsl(r-u.back())>1e-6L*max<ld>(1.0L,fabsl(r)))u.push_back(r);return u;
}

string summarizeObservedCalculus(const vector<ld>&x,const vector<ld>&y,const vector<ld>&zeros,const vector<ld>&turns,const vector<ld>&infl){
    if(x.size()<3||y.size()!=x.size())return{};
    vector<ld>s;for(size_t i=1;i<x.size();++i){ld dx=x[i]-x[i-1];if(fabsl(dx)>1e-18L)s.push_back((y[i]-y[i-1])/dx);}
    int pos=0,neg=0,flat=0;for(ld q:s){if(q>1e-9L)++pos;else if(q<-1e-9L)++neg;else ++flat;}
    string mono=(pos&&neg)?"non-monotonic":(pos?"increasing":(neg?"decreasing":"locally flat"));
    vector<ld>curv;for(size_t i=1;i+1<s.size()+1&&i+1<x.size();++i){ld h1=x[i]-x[i-1],h2=x[i+1]-x[i];if(h1<=0||h2<=0)continue;ld d1=s[i-1],d2=s[i];curv.push_back((d2-d1)/max<ld>(0.5L*(h1+h2),1e-12L));}
    int convex=0,concave=0;for(ld q:curv){if(q>1e-8L)++convex;else if(q<-1e-8L)++concave;}
    string curvature=(convex&&concave)?"mixed convex/concave":(convex?"mostly convex":(concave?"mostly concave":"near-zero curvature"));
    ostringstream os;os<<"Observed calculus: "<<mono<<" ("<<pos<<" increasing-slope intervals, "<<neg<<" decreasing-slope intervals, "<<flat<<" near-flat), "
      <<curvature<<" ("<<convex<<" positive-curvature, "<<concave<<" negative-curvature intervals), "
      <<turns.size()<<" turning point(s), "<<infl.size()<<" inflection candidate(s), "<<zeros.size()<<" numerical zero crossing(s).";
    if(!turns.empty()){os<<" Extrema candidates: ";for(size_t i=0;i<turns.size()&&i<12;++i){if(i)os<<",";os<<fmt(turns[i],4);}}
    if(!infl.empty()){os<<" Inflections: ";for(size_t i=0;i<infl.size()&&i<12;++i){if(i)os<<",";os<<fmt(infl[i],4);}}
    return os.str();
}

ld estimateExtremaPeriod(const vector<ld>&x,const vector<ld>&y){
    auto t=observedTurningPoints(x,y);if(t.size()<5)return numeric_limits<ld>::quiet_NaN();
    vector<ld>d;for(size_t i=1;i<t.size();++i){ld q=t[i]-t[i-1];if(q>1e-9L)d.push_back(q);}
    if(d.size()<4)return numeric_limits<ld>::quiet_NaN();
    ld half=quantile(d,.5L);if(!(half>0))return numeric_limits<ld>::quiet_NaN();
    // Alternating extrema correspond to roughly half a cycle.
    return 2.0L*half;
}


// Root-constrained regression. A detected zero is used as a structural factor
// so the fitted curve remains consistent with the observed crossing.
Model fitRootGuidedPolynomial(const vector<ld>& x,const vector<ld>& y,int degree,int rootsToUse=1){
    if(x.size()<10||x.size()!=y.size()||degree<0||rootsToUse<1||rootsToUse>3)return{};
    auto roots=numericRealRoots([&x,&y](ld z){
        if(z<=x.front())return y.front();
        if(z>=x.back())return y.back();
        auto it=lower_bound(x.begin(),x.end(),z);size_t j=(size_t)(it-x.begin());
        if(j==0)return y[0];
        ld q=(z-x[j-1])/(x[j]-x[j-1]);
        return y[j-1]*(1-q)+y[j]*q;
    },x.front(),x.back(),min(4096,max(128,sizeToInt(x.size()*24))));
    if(roots.empty())return{};
    vector<ld>cand=roots;
    if((int)cand.size()>8){sort(cand.begin(),cand.end(),[&](ld a,ld b){return fabsl(a-x.back())<fabsl(b-x.back());});cand.resize(8);sort(cand.begin(),cand.end());}
    vector<vector<ld>> sets;
    if(rootsToUse==1)for(ld r:cand)sets.push_back({r});
    else if(rootsToUse==2)for(size_t i=0;i<cand.size();++i)for(size_t j=i+1;j<cand.size();++j)sets.push_back({cand[i],cand[j]});
    else for(size_t i=0;i<cand.size();++i)for(size_t j=i+1;j<cand.size();++j)for(size_t k=j+1;k<cand.size();++k)sets.push_back({cand[i],cand[j],cand[k]});
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end()),mid=(xmin+xmax)/2,half=max<ld>((xmax-xmin)/2,1e-12L);
    ld best=numeric_limits<ld>::infinity();vector<ld>bc,br;
    for(auto&rs:sets){
        int p=degree+1;vector<vector<ld>>A(x.size(),vector<ld>(p));
        for(size_t i=0;i<x.size();++i){ld t=(x[i]-mid)/half,factor=1,q=1;for(ld r:rs)factor*=x[i]-r;for(int k=0;k<p;++k){A[i][k]=factor*q;q*=t;}}
        vector<ld>c;if(!leastSquaresQR(A,y,c))continue;
        ld sse=0;for(size_t i=0;i<x.size();++i){ld t=(x[i]-mid)/half,factor=1,q=1,pred=0;for(ld r:rs)factor*=x[i]-r;for(int k=0;k<p;++k){pred+=c[k]*factor*q;q*=t;}ld e=y[i]-pred;sse+=e*e;}
        if(sse<best){best=sse;bc=c;br=rs;}
    }
    if(bc.empty())return{};
    Model m;m.valid=true;m.name="Root-guided constrained polynomial";m.category="Root-aware polynomial / numerical analysis";m.complexity=degree+(int)br.size()+1;
    m.predict=[bc,br,mid,half](ld z){ld f=1,t=(z-mid)/half,q=1,s=0;for(ld r:br)f*=z-r;for(ld c:bc){s+=c*f*q;q*=t;}return s;};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    ostringstream os;os<<"P(x)*";for(size_t i=0;i<br.size();++i){if(i)os<<"*";os<<"(x-"<<fmt(br[i],6)<<")";}os<<", deg(P)="<<degree;m.formula=os.str();m.approxRealRoots=br;return m;
}

// ----------------------------- Exact algorithms -----------------
Model exactAffine(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<4||x.size()!=y.size())return{};
    ld dx=x[1]-x[0];if(fabsl(dx)<1e-18L)return{};ld b=(y[1]-y[0])/dx,a=y[0]-b*x[0];
    for(size_t i=2;i<y.size();++i)if(!approxEqual(y[i],a+b*x[i],1e-10L,1e-10L))return{};
    Model m;m.valid=m.exact=true;m.name="Exact affine relation";m.category="Exact curve relation";m.formula="y = "+fmt(a)+" + ("+fmt(b)+")x";m.predict=[a,b](ld z){return a+b*z;};m.fitted=y;m.nextValue=m.predict(x.back()+1);m.complexity=2;return m;
}

Model exactAP(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 5) return {};
    ld d = y[1] - y[0];
    for (size_t i = 2; i < y.size(); ++i)
        if (!approxEqual(y[i] - y[i - 1], d, 1e-10L, 1e-10L)) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Arithmetic progression";
    m.category = "Exact sequence algorithm";
    m.formula = "a_n = a_1 + (n-1)*" + fmt(d);
    m.predict = [d, y, x0=x.empty()?1.0L:x.front()](ld z) { return y.front() + (z-x0) * d; };
    m.fitted = y;
    m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
    m.complexity = 1;
    m.metrics.score = 100;
    m.note = "Exact validation: constant first difference across the supplied sequence.";
    return m;
}

Model exactGP(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 5 || approxEqual(y[0], 0)) return {};
    ld r = y[1] / y[0];
    for (size_t i = 2; i < y.size(); ++i)
        if (approxEqual(y[i - 1], 0) || !approxEqual(y[i] / y[i - 1], r, 1e-9L, 1e-9L)) return {};
    if (!isfinite(r)) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Geometric progression";
    m.category = "Exact sequence algorithm";
    m.formula = "a_n = a_1 * (" + fmt(r) + ")^(n-1)";
    m.predict = [r, y, x0=x.empty()?1.0L:x.front()](ld z) { return y.front() * powl(r, z-x0); };
    m.fitted = y;
    m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
    m.complexity = 2;
    m.metrics.score = 100;
    return m;
}

ld factorialLD(int n) {
    if (n < 0) return numeric_limits<ld>::quiet_NaN();
    ld r = 1;
    for (int i = 2; i <= n; ++i) {
        r *= i;
        if (!isfinite(r)) return r;
    }
    return r;
}

Model exactFactorial(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 5) return {};
    bool oneBased = true, zeroBased = true;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!approxEqual(y[i], factorialLD((int)i + 1), 1e-9L, 1e-9L)) oneBased = false;
        if (!approxEqual(y[i], factorialLD((int)i), 1e-9L, 1e-9L)) zeroBased = false;
    }
    if (!oneBased && !zeroBased) return {};
    bool z = zeroBased;
    Model m;
    m.valid = m.exact = true;
    m.name = z ? "Factorial (0-based)" : "Factorial";
    m.category = "Exact sequence algorithm";
    m.formula = z ? "a_n = (n-1)!" : "a_n = n!";
    m.predict = [z](ld pos) { return factorialLD((int)llroundl(pos) - (z ? 1 : 0)); };
    m.fitted = y;
    m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
    m.complexity = 2;
    return m;
}

Model exactFibonacci(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 6) return {};
    for (size_t i = 2; i < y.size(); ++i)
        if (!approxEqual(y[i], y[i - 1] + y[i - 2], 1e-10L, 1e-10L)) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Fibonacci-type recurrence";
    m.category = "Exact recurrence algorithm";
    m.formula = "a_n = a_(n-1) + a_(n-2)";
    m.predict = [y, lastX = (x.empty() ? (ld)y.size() : x.back())](ld z) {
        int steps = (int)llroundl(z - lastX);
        if (steps <= 0) return numeric_limits<ld>::quiet_NaN();
        ld a = y[y.size() - 2], b = y.back();
        for (int i = 0; i < steps; ++i) {
            ld c = a + b; a = b; b = c;
        }
        return b;
    };
    m.fitted = y;
    m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
    m.complexity = 2;
    return m;
}

long double nCrLD(int n, int r) {
    if (r < 0 || r > n) return 0;
    r = min(r, n - r);
    ld res = 1;
    for (int i = 1; i <= r; ++i) {
        res = res * (n - r + i) / i;
        if (!isfinite(res)) return res;
    }
    return res;
}

Model exactPowerFamily(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 6) return {};
    const auto n = sizeToInt(y.size());
    for (int p = 2; p <= 8; ++p) {
        bool ok = true;
        for (int i = 0; i < n; ++i)
            if (!approxEqual(y[i], powl((ld)i + 1, p), 1e-10L, 1e-10L)) { ok = false; break; }
        if (ok) {
            Model m;
            m.valid = m.exact = true;
            m.name = "Power sequence n^" + to_string(p);
            m.category = "Exact sequence algorithm";
            m.formula = "a_n = n^" + to_string(p);
            m.predict = [p](ld z) { return powl(z, p); };
            m.fitted = y;
            m.nextValue = m.predict(x.empty() ? n + 1 : x.back() + 1);
            m.complexity = p;
            return m;
        }
    }
    bool self = true;
    for (int i = 0; i < n; ++i)
        if (!approxEqual(y[i], powl((ld)i + 1, (ld)i + 1), 1e-8L, 1e-8L)) { self = false; break; }
    if (self) {
        Model m;
        m.valid = m.exact = true;
        m.name = "Self-power sequence";
        m.category = "Exact sequence algorithm";
        m.formula = "a_n = n^n";
        m.predict = [](ld z) { return powl(z, z); };
        m.fitted = y;
        m.nextValue = m.predict(x.empty() ? n + 1 : x.back() + 1);
        m.complexity = 3;
        return m;
    }
    return {};
}

Model exactCatalan(const vector<ld>&, const vector<ld>& y) {
    if (y.size() < 6) return {};
    for (int i = 0; i < (int)y.size(); ++i) {
        ld c = nCrLD(2 * i, i) / (i + 1);
        if (!approxEqual(y[i], c, 1e-9L, 1e-9L)) return {};
    }
    Model m;
    m.valid = m.exact = true;
    m.name = "Catalan numbers";
    m.category = "Exact combinatorial sequence";
    m.formula = "C_n = binomial(2n,n)/(n+1), n starts at 0";
    m.predict = [](ld z) { int n = (int)llroundl(z); return nCrLD(2 * n, n) / (n + 1); };
    m.fitted = y;
    m.nextValue = m.predict((ld)y.size());
    m.complexity = 3;
    return m;
}

Model exactBell(const vector<ld>&, const vector<ld>& y) {
    if (y.size() < 5) return {};
    const auto n = sizeToInt(y.size());
    vector<ld> b(n + 1, 0);
    b[0] = 1;
    for (int i = 1; i <= n; ++i) {
        ld s = 0;
        for (int k = 0; k < i; ++k) s += nCrLD(i - 1, k) * b[k];
        b[i] = s;
    }
    for (int i = 0; i < n; ++i) if (!approxEqual(y[i], b[i], 1e-8L, 1e-8L)) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Bell numbers";
    m.category = "Exact combinatorial sequence";
    m.formula = "B_n = sum_{k=0}^{n-1} binomial(n-1,k) B_k";
    m.predict = [](ld z) {
        int q = (int)llroundl(z);
        vector<ld> bb(q + 1, 0); bb[0] = 1;
        for (int i = 1; i <= q; ++i) {
            ld s = 0;
            for (int k = 0; k < i; ++k) s += nCrLD(i - 1, k) * bb[k];
            bb[i] = s;
        }
        return bb[q];
    };
    m.fitted = y;
    m.nextValue = m.predict(n);
    m.complexity = 3;
    return m;
}

bool isPrimeLL(long long n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (long long d = 3; d <= n / d; d += 2) if (n % d == 0) return false;
    return true;
}

Model exactPrimeSequence(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 6 || !isInteger(y[0])) return {};
    long long first = llroundl(y[0]);
    if (!isPrimeLL(first)) return {};
    vector<long long> primes;
    for (long long v = 2; (int)primes.size() < 60000; ++v)
        if (isPrimeLL(v)) primes.push_back(v);
    auto it = find(primes.begin(), primes.end(), first);
    if (it == primes.end()) return {};
    size_t st = (size_t)(it - primes.begin());
    if (st + y.size() >= primes.size()) return {};
    for (size_t i = 0; i < y.size(); ++i)
        if (!approxEqual(y[i], (ld)primes[st + i], 1e-12L, 1e-12L)) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Consecutive prime sequence";
    m.category = "Exact number-theoretic sequence";
    m.formula = "Consecutive primes starting at " + to_string(first);
    m.predict = [st, primes](ld z) {
        long long pos = llroundl(z) - 1;
        if (pos < 0 || (size_t)pos >= primes.size() - st) return numeric_limits<ld>::quiet_NaN();
        return (ld)primes[st + (size_t)pos];
    };
    m.fitted = y;
    m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
    m.complexity = 2;
    return m;
}

Model exactPolygonal(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 6) return {};
    static const char* names[] = {"", "", "triangular", "square", "pentagonal", "hexagonal", "heptagonal", "octagonal", "nonagonal", "decagonal"};
    for (int s = 3; s <= 10; ++s) {
        bool ok = true;
        for (int i = 0; i < (int)y.size(); ++i) {
            ld n = i + 1;
            ld p = (n * n * (s - 2) - n * (s - 4)) / 2;
            if (!approxEqual(y[i], p, 1e-10L, 1e-10L)) { ok = false; break; }
        }
        if (ok) {
            Model m;
            m.valid = m.exact = true;
            m.name = string(names[s]) + " polygonal sequence";
            m.category = "Exact geometric sequence";
            m.formula = "P_s(n)=(n^2(s-2)-n(s-4))/2";
            m.predict = [s](ld n) { return (n * n * (s - 2) - n * (s - 4)) / 2; };
            m.fitted = y;
            m.nextValue = m.predict(x.empty() ? (ld)y.size() + 1 : x.back() + 1);
            m.complexity = 2;
            return m;
        }
    }
    return {};
}

Model exactRecurrence2(const vector<ld>& x, const vector<ld>& y) {
    if (y.size() < 8) return {};
    vector<vector<ld>> A; vector<ld> b;
    for (size_t i = 2; i < y.size(); ++i) { A.push_back({y[i - 1], y[i - 2]}); b.push_back(y[i]); }
    vector<ld> c;
    if (!leastSquaresQR(A, b, c) || c.size() != 2) return {};
    for (size_t i = 2; i < y.size(); ++i)
        if (!approxEqual(y[i], c[0] * y[i - 1] + c[1] * y[i - 2], 1e-8L, 1e-8L)) return {};
    ld lastX = x.empty() ? (ld)y.size() : x.back();
    Model m;
    m.valid = m.exact = true;
    m.name = "2nd-order linear recurrence";
    m.category = "Exact recurrence algorithm";
    m.formula = "a_n=" + fmt(c[0]) + "a_(n-1)+" + fmt(c[1]) + "a_(n-2)";
    m.predict = [p = c[0], q = c[1], y, lastX](ld z) {
        int steps = (int)llroundl(z - lastX);
        if (steps <= 0) return numeric_limits<ld>::quiet_NaN();
        ld a = y[y.size() - 2], b = y.back();
        for (int i = 0; i < steps; ++i) {
            ld c2 = p * b + q * a; a = b; b = c2;
            if (!isfinite(b)) return numeric_limits<ld>::quiet_NaN();
        }
        return b;
    };
    m.fitted = y;
    m.nextValue = m.predict(lastX + 1);
    m.complexity = 2;
    return m;
}

Model exactPeriodic(const vector<ld>& x, const vector<ld>& y) {
    (void)x;
    const auto n = sizeToInt(y.size());
    if (n < 8) return {};
    for (int p = 1; p <= n / 3; ++p) {
        bool ok = true;
        for (int i = p; i < n; ++i)
            if (!approxEqual(y[i], y[i % p], 1e-10L, 1e-10L)) { ok = false; break; }
        if (ok) {
            Model m;
            m.valid = m.exact = true;
            m.name = "Periodic repetition (period " + to_string(p) + ")";
            m.category = "Exact cyclic algorithm";
            m.formula = "a_n = a_((n-1) mod " + to_string(p) + "+1)";
            const ld x0=x.empty()?1.0L:x.front();
            m.predict = [p, y, x0](ld z) {
                long long idx = llroundl(z-x0);
                if (idx < 0) return numeric_limits<ld>::quiet_NaN();
                return y[(size_t)(idx % p)];
            };
            m.fitted = y;
            m.nextValue = m.predict(x0+n);
            m.complexity = 1;
            return m;
        }
    }
    return {};
}

Model exactInterleaved(const vector<ld>& x, const vector<ld>& y) {
    const auto n = sizeToInt(y.size());
    if (n < 10) return {};
    for (int k = 2; k <= 6; ++k) {
        if (n < k * 5) continue;
        vector<Model> sub(k);
        bool all = true;
        for (int lane = 0; lane < k; ++lane) {
            vector<ld> v;
            for (int i = lane; i < n; i += k) v.push_back(y[i]);
            Model m = exactAP({}, v);
            if (!m.valid) m = exactGP({}, v);
            if (!m.valid) m = exactPowerFamily({}, v);
            if (!m.valid) m = exactFibonacci({}, v);
            if (!m.valid) { all = false; break; }
            sub[lane] = m;
        }
        if (!all) continue;
        Model m;
        m.valid = m.exact = true;
        m.name = "Interleaved " + to_string(k) + "-sequence pattern";
        m.category = "Interleaved sequence algorithm";
        ostringstream os;
        os << "position mod " << k << " uses independent exact rules: ";
        for (int lane = 0; lane < k; ++lane) {
            if (lane) os << " | ";
            os << "lane " << lane << ": " << sub[lane].formula;
        }
        m.formula = os.str();
        const ld x0=x.empty()?1.0L:x.front();
        m.predict = [sub, k, x0](ld z) {
            long long pos = llroundl(z-x0);
            if (pos < 0) return numeric_limits<ld>::quiet_NaN();
            int lane = (int)(pos % k);
            ld subPos = (ld)(pos / k + 1);
            return sub[lane].predict(subPos);
        };
        m.fitted = y;
        m.nextValue = m.predict(x0 + n);
        m.complexity = k;
        return m;
    }
    (void)x;
    return {};
}

Model exactStaircase(const vector<ld>& x,const vector<ld>& y){
    (void)x;
    const int n=sizeToInt(y.size());
    if(n<8) return {};
    vector<ld>levels; vector<int>runs;
    for(ld v:y){
        if(!levels.empty() && approxEqual(v,levels.back(),1e-10L,1e-10L)) runs.back()++;
        else {levels.push_back(v);runs.push_back(1);}
    }
    if(levels.size()<3) return {};
    if(runs.front()<2) return {};
    for(int r:runs) if(r!=runs.front()) return {};
    const int k=sizeToInt(levels.size());
    bool ap=true,gp=true;ld d=levels[1]-levels[0];
    for(int i=2;i<k;++i) if(!approxEqual(levels[i]-levels[i-1],d,1e-9L,1e-9L)){ap=false;break;}
    ld ratio=(fabsl(levels[0])>1e-18L?levels[1]/levels[0]:0);
    if(!isfinite(ratio)||fabsl(levels[0])<1e-18L) gp=false;
    if(gp) for(int i=2;i<k;++i) if(!approxEqual(levels[i]/levels[i-1],ratio,1e-9L,1e-9L)){gp=false;break;}
    if(!ap&&!gp) return {};
    const int run=runs.front();
    Model m; m.valid=m.exact=true; m.name=ap?"Exact arithmetic staircase / stepped progression":"Exact geometric staircase / stepped progression";
    m.category="Exact piecewise staircase algorithm"; m.complexity=ap?2:3;
    if(ap) m.formula="levels follow arithmetic progression; each level is held for "+to_string(run)+" observations";
    else m.formula="levels follow geometric progression; each level is held for "+to_string(run)+" observations";
    m.predict=[levels,run,ap,d,ratio](ld z){
        long long idx=llroundl(z)-1; if(idx<0) return numeric_limits<ld>::quiet_NaN();
        long long lev=idx/run;
        if(ap) return levels.front()+((ld)lev)*d;
        return levels.front()*powl(ratio,(ld)lev);
    };
    m.fitted.reserve(y.size());for(size_t i=0;i<y.size();++i)m.fitted.push_back(y[i]);
    m.nextValue=m.predict((ld)n+1);
    m.note="Detected repeated plateaus whose levels themselves follow a deterministic progression; the next plateau level is forecast rather than holding the last value.";
    return m;
}

Model fitRobustStaircaseProgression(const vector<ld>&x,const vector<ld>&y){
    if(x.size()!=y.size()||y.size()<15||!isUnitSpacedGrid(x))return{};
    const int n=sizeToInt(y.size());
    vector<ld>d;d.reserve(n-1);for(int i=1;i<n;++i)d.push_back(y[i]-y[i-1]);
    vector<ld>ad;for(ld q:d)ad.push_back(fabsl(q));
    ld diffNoise=max({quantile(ad,.35L),medianAbs(d)*0.5L,1e-10L});
    ld levelScale=max<ld>(robustScale(y),1e-9L);
    ld threshold=max<ld>(4.0L*diffNoise,0.015L*levelScale);
    vector<int>starts={0};
    for(int i=1;i<n;++i)if(fabsl(d[i-1])>threshold)starts.push_back(i);
    if(starts.size()<5)return{};
    vector<int>lens;for(size_t i=1;i<starts.size();++i)lens.push_back(starts[i]-starts[i-1]);lens.push_back(n-starts.back());
    ld ml=meanOf(vector<ld>(lens.begin(),lens.end())), sd=0;for(int q:lens)sd+=(q-ml)*(q-ml);sd=sqrtl(sd/max<size_t>(1,lens.size()));
    if(ml<1.5L||sd/max<ld>(ml,1e-9L)>0.30L)return{};
    vector<ld>levels;vector<int>runs;levels.reserve(starts.size());runs.reserve(lens.size());
    for(size_t j=0;j<starts.size();++j){int lo=starts[j],hi=(j+1<starts.size()?starts[j+1]:n);vector<ld>seg(y.begin()+lo,y.begin()+hi);levels.push_back(quantile(seg,.5L));runs.push_back(hi-lo);}
    if(levels.size()<4)return{};
    bool ap=true;ld diff=levels[1]-levels[0];for(size_t i=2;i<levels.size();++i)if(fabsl((levels[i]-levels[i-1])-diff)>max<ld>(0.12L*max<ld>(fabsl(diff),1e-6L),0.08L*diffNoise)){ap=false;break;}
    bool gp=levels.front()!=0;ld ratio=gp?levels[1]/levels[0]:0;if(!isfinite(ratio)||fabsl(levels[0])<1e-8L)gp=false;
    if(gp)for(size_t i=2;i<levels.size();++i)if(fabsl((levels[i]/max<ld>(levels[i-1],1e-12L))-ratio)>0.12L*max<ld>(fabsl(ratio),1.0L)){gp=false;break;}
    if(!ap&&!gp)return{};
    int run=(int)llround(quantile(vector<ld>(lens.begin(),lens.end()),.5L));if(run<1)return{};
    Model m;m.valid=true;m.exact=false;m.name=ap?"Robust arithmetic staircase progression":"Robust geometric staircase progression";m.category="Noise-tolerant piecewise staircase / stepper model";m.complexity=ap?3:4;
    if(ap)m.formula="plateau levels follow an arithmetic progression; robust median run length="+to_string(run);else m.formula="plateau levels follow a geometric progression; robust median run length="+to_string(run);
    const ld x0=x.front();
    const ld stepLevel=levels.size()>1?levels[1]-levels[0]:0.0L;
    m.predict=[x0,levels,run,ap,stepLevel,ratio](ld z){long long idx=llroundl(z-x0);if(idx<0)return numeric_limits<ld>::quiet_NaN();long long k=idx/run;if(ap)return levels.front()+(ld)k*stepLevel;return levels.front()*powl(ratio,(ld)k);};
    m.fitted.reserve(y.size());for(ld z:x)m.fitted.push_back(m.predict(z));m.nextValue=m.predict(x.back()+1);
    m.note="Detected step events from first-difference spikes, then inferred a regular plateau duration and progression of plateau levels. This is robust to small within-plateau noise.";
    return m;
}

Model exactStepFunction(const vector<ld>& x, const vector<ld>& y) {
    // Zero-order-hold / staircase signal: the data consists of exact
    // constant-value plateaus, as in a control-system step response, a
    // quantized sensor, or a discrete state machine. This is a very common
    // "signals & systems" shape that was previously only ever approximated
    // by change-point regression, which can miss the reliability gate on
    // short runs even when the underlying data is perfectly clean.
    (void)x;
    const auto n = sizeToInt(y.size());
    if (n < 4) return {};
    vector<pair<ld,int>> segments;
    for (int i = 0; i < n; ++i) {
        if (!segments.empty() && approxEqual(y[i], segments.back().first, 1e-10L, 1e-10L)) segments.back().second++;
        else segments.push_back({y[i], 1});
    }
    if (segments.size() < 2) return {}; // a fully constant series is handled elsewhere
    int repeatedPoints = n - (int)segments.size(); // points that extended an existing plateau
    // Require genuine repeated-value evidence (at least two "held" samples) and
    // that plateaus are not just isolated one-off coincidences -- otherwise a
    // generic, never-repeating sequence would trivially "match" with runs of
    // length 1 each, which is not a step signal at all.
    if (repeatedPoints < 2 || (int)segments.size() > n/2 + 1) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Piecewise-constant step signal (zero-order hold)";
    m.category = "Exact signal/systems algorithm";
    ostringstream os;
    os << "Exact plateaus: ";
    for (size_t i = 0; i < segments.size(); ++i) { if (i) os << " -> "; os << fmt(segments[i].first) << " x" << segments[i].second; }
    m.formula = os.str();
    ld lastLevel = segments.back().first;
    m.predict = [lastLevel](ld) { return lastLevel; };
    m.fitted = y;
    m.nextValue = lastLevel;
    m.complexity = (int)segments.size();
    m.note = "Extrapolated by zero-order hold: the exact time of the next step is not inferable from the levels alone, so the last observed plateau is held forward.";
    return m;
}

Model exactEventualCycle(const vector<ld>& x, const vector<ld>& y) {
    // Graph-theoretic "rho shape" detection: this is the structure Floyd's
    // and Brent's tortoise-and-hare cycle-finding algorithms are built to
    // exploit for an iterated function -- a non-repeating tail of length mu
    // followed by a cycle of length lambda that repeats forever after.
    // exactPeriodic above only covers the special case mu=0 (periodic from
    // the very first term); this generalizes it. That generalization is
    // exactly the tool needed for any iterated/state-machine process that
    // only settles into a cycle after a transient -- hash-chain iteration,
    // modular recurrences, or a sequence of states visited while exploring
    // a permutation-group state graph (e.g. a puzzle's move sequence
    // eventually revisiting a state). It does not search for or solve such
    // a graph; it only recognizes an eventual repeat once the states/values
    // that occurred are given as data.
    (void)x;
    const auto n = sizeToInt(y.size());
    if (n < 9) return {};
    int bestMu = -1, bestLambda = -1;
    for (int lambda = 1; lambda <= n / 3 && bestLambda < 0; ++lambda) {
        for (int mu = 0; mu + 3 * lambda <= n; ++mu) {
            bool ok = true;
            for (int i = mu; i + lambda < n && ok; ++i)
                if (!approxEqual(y[i], y[i + lambda], 1e-10L, 1e-10L)) ok = false;
            if (ok) { bestMu = mu; bestLambda = lambda; break; }
        }
    }
    if (bestLambda < 0 || bestMu == 0) return {}; // mu==0 belongs to exactPeriodic
    Model m;
    m.valid = m.exact = true;
    m.name = "Eventual cycle (tail " + to_string(bestMu) + " + period " + to_string(bestLambda) + ")";
    m.category = "Exact graph-theoretic algorithm (functional-graph cycle)";
    m.formula = "a_n = a_(n-" + to_string(bestLambda) + ") once past position " + to_string(bestMu) + " (a non-repeating tail precedes the cycle)";
    int mu = bestMu, lambda = bestLambda;
    m.predict = [y, mu, lambda](ld z) {
        long long idx = llroundl(z) - 1;
        if (idx < 0) return numeric_limits<ld>::quiet_NaN();
        if (idx < mu) return y[(size_t)idx];
        long long off = (idx - mu) % lambda;
        return y[(size_t)(mu + off)];
    };
    m.fitted = y;
    m.nextValue = m.predict((ld)n + 1);
    m.complexity = bestMu + bestLambda;
    m.note = "Detected via tortoise-and-hare style tail+cycle matching over the observed data (a functional-graph / rho-shape structure).";
    return m;
}

Model exactPermutation(const vector<ld>& y) {
    if (y.size() < 8) return {};
    vector<long long> v;
    for (ld z : y) { if (!isInteger(z)) return {}; v.push_back(llroundl(z)); }
    set<long long> s(v.begin(), v.end());
    if (s.size() != v.size()) return {};
    long long mn = *min_element(v.begin(), v.end()), mx = *max_element(v.begin(), v.end());
    if (mx - mn + 1 != (long long)v.size()) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Permutation / shuffled contiguous integers";
    m.category = "Order-statistical structure";
    m.formula = "All integers in [" + to_string(mn) + "," + to_string(mx) + "] occur exactly once; order is not a deterministic rule.";
    m.predict = [](ld) { return numeric_limits<ld>::quiet_NaN(); };
    m.complexity = 0;
    m.note = "Set membership is exact, but membership alone does not determine the next ordering.";
    return m;
}

Model exactDistinctIntegerSet(const vector<ld>& y) {
    if (y.size() < 10) return {};
    vector<long long> v;
    for (ld z : y) { if (!isInteger(z)) return {}; v.push_back(llroundl(z)); }
    set<long long> seen(v.begin(), v.end());
    if (seen.size() != v.size()) return {};
    long long mn = *min_element(v.begin(), v.end()), mx = *max_element(v.begin(), v.end());
    long long span = mx - mn + 1;
    if (span < (long long)y.size()) return {};
    ld coverage = (ld)y.size() / max<ld>(1, span);
    if (coverage < 0.50L) return {};
    Model m;
    m.valid = m.exact = true;
    m.name = "Distinct integer subset / sampled ordering";
    m.category = "Order / sampling structure";
    m.formula = "Distinct integers in [" + to_string(mn) + "," + to_string(mx) + "]; coverage=" + fmt(coverage * 100, 3) + "%";
    m.predict = [](ld) { return numeric_limits<ld>::quiet_NaN(); };
    long long missing = span - (long long)y.size();
    m.note = "No deterministic next term follows from distinct-set membership alone. " + to_string(missing) + " integer value(s) are absent from the observed range.";
    return m;
}

// ----------------------------- PRNG algorithms ------------------
bool modInverseULL(uint64_t a, uint64_t m, uint64_t& inv) {
    if (m == 0 || std::gcd(a, m) != 1) return false;
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)a;
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = t - q * newt; t = newt; newt = tmp;
        tmp = r - q * newr; r = newr; newr = tmp;
    }
    if (r != 1) return false;
    t %= (int64_t)m;
    if (t < 0) t += (int64_t)m;
    inv = (uint64_t)t;
    return true;
}

Model detectLCG(const vector<ld>& x, const vector<ld>& y) {
    (void)x;
    if (y.size() < 6) return {};
    vector<uint64_t> v;
    for (ld q : y) {
        if (!isInteger(q) || q < 0 || q > (ld)numeric_limits<uint32_t>::max()) return {};
        v.push_back((uint64_t)llroundl(q));
    }
    uint64_t mx = *max_element(v.begin(), v.end());
    vector<uint64_t> mods;
    auto addMod = [&](uint64_t m) {
        if (m > mx && m >= 3 && m <= (1ULL << 32) && find(mods.begin(), mods.end(), m) == mods.end()) mods.push_back(m);
    };
    addMod(mx + 1);
    for (int k = 3; k <= 32; ++k) addMod(1ULL << k);
    if (mx + 2 <= (1ULL << 32)) addMod(mx + 2);

    for (uint64_t m : mods) {
        uint64_t d1 = v[1] >= v[0] ? (v[1] - v[0]) % m : (m - ((v[0] - v[1]) % m)) % m;
        uint64_t d2 = v[2] >= v[1] ? (v[2] - v[1]) % m : (m - ((v[1] - v[2]) % m)) % m;
        uint64_t inv = 0;
        if (!modInverseULL(d1, m, inv)) continue;
        uint64_t a = (uint64_t)(d2 * inv % m);
        uint64_t c = (uint64_t)(((__int128_t)v[1] - a * v[0]) % m);
        if ((int64_t)c < 0) c += m;
        bool ok = true;
        for (size_t i = 0; i + 1 < v.size(); ++i) {
            uint64_t got = (uint64_t)((a * v[i] + c) % m);
            if (got != v[i + 1]) { ok = false; break; }
        }
        if (!ok) continue;
        uint64_t last = v.back();
        Model mod;
        mod.valid = mod.exact = true;
        mod.name = "Linear congruential generator (LCG)";
        mod.category = "Deterministic pseudo-random algorithm";
        mod.formula = "x_(n+1) = (" + to_string(a) + "*x_n + " + to_string(c) + ") mod " + to_string(m);
        mod.predict = [a, c, m, last, n = (int)y.size()](ld z) {
            int steps = (int)llroundl(z - n);
            if (steps <= 0) return numeric_limits<ld>::quiet_NaN();
            uint64_t cur = last;
            for (int i = 0; i < steps; ++i) cur = (uint64_t)((a * cur + c) % m);
            return (ld)cur;
        };
        mod.fitted = y;
        mod.nextValue = mod.predict((ld)y.size() + 1);
        mod.complexity = 3;
        mod.note = "Exact recurrence matches the observed integer stream under the inferred LCG parameters.";
        return mod;
    }
    return {};
}

uint32_t xorshift32(uint32_t x) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

Model detectXorShift32(const vector<ld>& x, const vector<ld>& y) {
    (void)x;
    if (y.size() < 6) return {};
    vector<uint32_t> v;
    for (ld q : y) {
        if (!isInteger(q) || q < 0 || q > (ld)numeric_limits<uint32_t>::max()) return {};
        v.push_back((uint32_t)llroundl(q));
    }
    for (size_t i = 0; i + 1 < v.size(); ++i)
        if (xorshift32(v[i]) != v[i + 1]) return {};
    uint32_t last = v.back();
    Model m;
    m.valid = m.exact = true;
    m.name = "xorshift32 PRNG";
    m.category = "Deterministic pseudo-random algorithm";
    m.formula = "x ^= x<<13; x ^= x>>17; x ^= x<<5";
    m.predict = [last, n = (int)y.size()](ld z) {
        int steps = (int)llroundl(z - n);
        if (steps <= 0) return numeric_limits<ld>::quiet_NaN();
        uint32_t cur = last;
        for (int i = 0; i < steps; ++i) cur = xorshift32(cur);
        return (ld)cur;
    };
    m.fitted = y;
    m.nextValue = m.predict((ld)y.size() + 1);
    m.complexity = 3;
    m.note = "Exact xorshift32 state transition matches every observed pair.";
    return m;
}

// ----------------------------- Advanced model engine ----------------
struct FamilySpec { string family; int degree = 1; ld param = 0, param2 = 0; };
bool isUnitSpacedGrid(const vector<ld>& x);

struct BasisTerm {
    string name;
    function<ld(ld)> f;
    bool isIntercept = false;
};

enum class Regularizer {
    Ridge,
    Lasso,
    ElasticNet,
    Huber
};

string regularizerName(Regularizer r) {
    switch (r) {
        case Regularizer::Ridge: return "Ridge";
        case Regularizer::Lasso: return "Lasso";
        case Regularizer::ElasticNet: return "Elastic Net";
        case Regularizer::Huber: return "Huber IRLS";
    }
    return "Regularized";
}

ld softThreshold(ld z, ld g) {
    if (z > g) return z - g;
    if (z < -g) return z + g;
    return 0;
}

bool weightedLeastSquares(const vector<vector<ld>>& A,
                          const vector<ld>& y,
                          const vector<ld>& weights,
                          vector<ld>& beta) {
    if (A.empty() || A.size() != y.size() || weights.size() != y.size()) return false;
    vector<vector<ld>> Aw(A.size());
    vector<ld> yw(y.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ld w = sqrtl(max<ld>(weights[i], 0));
        Aw[i].resize(A[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) Aw[i][j] = A[i][j] * w;
        yw[i] = y[i] * w;
    }
    return leastSquaresQR(Aw, yw, beta);
}

bool ridgeSolve(const vector<vector<ld>>& A,
                const vector<ld>& y,
                ld lambda,
                vector<ld>& beta,
                bool penalizeIntercept = false) {
    if (A.empty() || A.size() != y.size() || A[0].empty()) return false;
    const int p = sizeToInt(A[0].size());
    for (const auto& row : A) if ((int)row.size() != p) return false;

    vector<vector<ld>> aug = A;
    vector<ld> by = y;

    // Ridge is represented as extra sqrt(lambda) rows.
    // The intercept is normally left unpenalized.
    const int startPenalty = penalizeIntercept ? 0 : 1;
    for (int j = startPenalty; j < p; ++j) {
        vector<ld> row(p, 0);
        row[j] = sqrtl(max<ld>(lambda, 0));
        aug.push_back(std::move(row));
        by.push_back(0);
    }
    return leastSquaresQR(aug, by, beta);
}

bool lassoCoordinateDescent(const vector<vector<ld>>& A,
                            const vector<ld>& y,
                            ld lambda,
                            ld l1Ratio,
                            vector<ld>& beta,
                            int maxIter = 500) {
    if (A.empty() || A.size() != y.size() || A[0].empty()) return false;
    const int n = sizeToInt(A.size()), p = sizeToInt(A[0].size());
    beta.assign(p, 0);
    l1Ratio = clamp(l1Ratio, 0.0L, 1.0L);
    vector<ld> colNorm(p, 0);
    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i) colNorm[j] += A[i][j] * A[i][j];
    if (p > 0) {
        ld meanY = meanOf(y);
        beta[0] = meanY;
    }

    vector<ld> residual(n, 0);
    for (int i = 0; i < n; ++i) residual[i] = y[i] - (p ? beta[0] : 0);

    for (int iter = 0; iter < maxIter; ++iter) {
        ld maxChange = 0;
        for (int j = 1; j < p; ++j) {
            if (colNorm[j] < 1e-18L) continue;
            ld rho = 0;
            for (int i = 0; i < n; ++i) rho += A[i][j] * (residual[i] + A[i][j] * beta[j]);
            ld denom = colNorm[j] / n + (1.0L - l1Ratio) * lambda;
            ld next = softThreshold(rho / n, l1Ratio * lambda) / max<ld>(denom, 1e-18L);
            ld delta = next - beta[j];
            if (delta != 0) {
                for (int i = 0; i < n; ++i) residual[i] -= A[i][j] * delta;
                beta[j] = next;
                maxChange = max(maxChange, fabsl(delta));
            }
        }
        // Update intercept exactly.
        if (p > 0) {
            ld meanResidual = meanOf(residual);
            beta[0] += meanResidual;
            for (int i = 0; i < n; ++i) residual[i] -= meanResidual;
        }
        if (maxChange < 1e-10L) break;
    }
    return all_of(beta.begin(), beta.end(), [](ld v) { return isfinite(v); });
}

bool fitRegularizedMatrix(const vector<vector<ld>>& A,
                          const vector<ld>& y,
                          Regularizer reg,
                          ld lambda,
                          vector<ld>& beta,
                          ld l1Ratio = 0.5L,
                          int huberIterations = 6) {
    if (A.empty() || A.size() != y.size()) return false;

    if (reg == Regularizer::Ridge) {
        return ridgeSolve(A, y, lambda, beta, false);
    }

    if (reg == Regularizer::Lasso || reg == Regularizer::ElasticNet) {
        return lassoCoordinateDescent(A, y, lambda, reg == Regularizer::Lasso ? 1.0L : l1Ratio, beta);
    }

    // Huber IRLS + mild ridge stabilization.
    beta.assign(A[0].size(), 0);
    vector<ld> weights(y.size(), 1.0L);
    for (int iter = 0; iter < huberIterations; ++iter) {
        if (!weightedLeastSquares(A, y, weights, beta)) return false;
        vector<ld> resid(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            ld p = 0;
            for (size_t j = 0; j < beta.size(); ++j) p += A[i][j] * beta[j];
            resid[i] = y[i] - p;
        }
        ld scale = max<ld>(1.4826L * medianAbs(resid), 1e-10L);
        const ld delta = 1.345L * scale;
        ld change = 0;
        for (size_t i = 0; i < weights.size(); ++i) {
            ld old = weights[i];
            ld a = fabsl(resid[i]);
            weights[i] = a <= delta ? 1.0L : delta / max(a, 1e-18L);
            change = max(change, fabsl(old - weights[i]));
        }
        if (change < 1e-7L) break;
    }
    return all_of(beta.begin(), beta.end(), [](ld v) { return isfinite(v); });
}

vector<BasisTerm> makeBasisLibrary(const vector<ld>& x) {
    vector<BasisTerm> t;
    if (x.empty()) return t;
    ld xmin = *min_element(x.begin(), x.end());
    ld xmax = *max_element(x.begin(), x.end());
    ld span = max<ld>(xmax - xmin, 1e-9L);
    ld mid = (xmin + xmax) / 2.0L;
    ld invHalf = 2.0L / span;

    auto u = [xmin, span](ld z) { return max<ld>(0, (z - xmin) / span); };
    auto tn = [mid, invHalf](ld z) { return clamp((z - mid) * invHalf, -6.0L, 6.0L); };
    auto safeExp = [](ld q) {
        q = clamp(q, -25.0L, 25.0L);
        return expl(q);
    };
    auto safeLog = [u](ld z) { return log1pl(max<ld>(0, u(z))); };

    t.push_back({"1", [](ld) { return 1.0L; }, true});
    t.push_back({"t", tn});
    t.push_back({"t^2", [tn](ld z) { ld v=tn(z); return v*v; }});
    t.push_back({"t^3", [tn](ld z) { ld v=tn(z); return v*v*v; }});
    t.push_back({"t^4", [tn](ld z) { ld v=tn(z); return v*v*v*v; }});
    t.push_back({"t^5", [tn](ld z) { ld v=tn(z); return v*v*v*v*v; }});
    t.push_back({"t^6", [tn](ld z) { ld v=tn(z); return v*v*v*v*v*v; }});

    // Raw-domain transforms are important because many real formulas are
    // defined directly in x rather than in a normalized coordinate.
    if (xmin > 0) {
        t.push_back({"ln(x)", [](ld z){return z>0?logl(z):0.0L;}});
    }
    const vector<ld> rawExpRates = {-0.20L,-0.10L,-0.05L,-0.025L,0.025L,0.05L,0.10L,0.20L};
    for (ld r : rawExpRates) {
        string nm="exp("+fmt(r,3)+"*(x-x0))";
        t.push_back({nm,[xmin,r](ld z){return expl(clamp(r*(z-xmin),-25.0L,25.0L));}});
    }
    const vector<ld> rawFreqs = {0.05L,0.10L,0.20L,0.25L,0.40L,0.50L,0.75L,1.0L,1.5L,2.0L};
    for (ld w : rawFreqs) {
        t.push_back({"sin("+fmt(w,3)+"x)",[w](ld z){return sinl(w*z);}});
        t.push_back({"cos("+fmt(w,3)+"x)",[w](ld z){return cosl(w*z);}});
    }
    t.push_back({"1/t", [tn](ld z) { ld v=tn(z); return 1.0L / copysignl(max<ld>(fabsl(v), 0.08L), v == 0 ? 1.0L : v); }});
    t.push_back({"1/t^2", [tn](ld z) { ld v=tn(z); v=copysignl(max<ld>(fabsl(v),0.08L),v==0?1.0L:v); return 1.0L/(v*v); }});
    t.push_back({"sqrt(u)", [u](ld z) { return sqrtl(max<ld>(u(z),0)); }});
    t.push_back({"log1p(u)", safeLog});
    t.push_back({"log1p(|t|)", [tn](ld z){return log1pl(fabsl(tn(z)));}});
    t.push_back({"abs(t)", [tn](ld z){return fabsl(tn(z));}});
    t.push_back({"sign(t)", [tn](ld z){ld v=tn(z);return v>0?1.0L:(v<0?-1.0L:0.0L);}});
    t.push_back({"tanh(t)", [tn](ld z){return tanhl(tn(z));}});
    t.push_back({"sigmoid(t)", [tn](ld z){ld q=clamp(-tn(z),-25.0L,25.0L);return 1.0L/(1.0L+expl(q));}});

    // Curvature/oscillation features in the original x-domain. These are
    // important for functions whose phase is not linear in x, e.g. sin(x^2),
    // Fresnel-like curves, chirps and radially symmetric compositions.
    const vector<ld> rawQuadraticFreqs = {0.0625L,0.125L,0.25L,0.5L,0.75L,1.0L,1.5L,2.0L,3.0L,4.0L};
    t.push_back({"x^2", [](ld z){return z*z;}});
    t.push_back({"x^3", [](ld z){return z*z*z;}});
    for(ld w:rawQuadraticFreqs){
        t.push_back({"sin("+fmt(w,4)+"x^2)",[w](ld z){return sinl(w*z*z);}});
        t.push_back({"cos("+fmt(w,4)+"x^2)",[w](ld z){return cosl(w*z*z);}});
    }
    // Gaussian/chirp families and nested analytic compositions. They remain
    // bounded, deterministic features rather than arbitrary expression trees.
    const vector<ld> gaussRates={0.25L,0.5L,1.0L,2.0L,4.0L};
    for(ld a:gaussRates){
        auto g=[a](ld z){return expl(clamp(-a*z*z,-40.0L,0.0L));};
        t.push_back({"exp(-"+fmt(a,3)+"x^2)",g});
        t.push_back({"sin(exp(-"+fmt(a,3)+"x^2))",[g](ld z){return sinl(g(z));}});
        t.push_back({"cos(exp(-"+fmt(a,3)+"x^2))",[g](ld z){return cosl(g(z));}});
        t.push_back({"tanh(exp(-"+fmt(a,3)+"x^2))",[g](ld z){return tanhl(g(z));}});
        t.push_back({"tanh(cos(exp(-"+fmt(a,3)+"x^2)))",[g](ld z){return tanhl(cosl(g(z)));}});
        t.push_back({"tanh(cos(log(sin(exp(-"+fmt(a,3)+"x^2)))))",[g](ld z){
            ld q=sinl(g(z)); if(q<=1e-14L) return 0.0L; return tanhl(cosl(logl(q)));
        }});
        t.push_back({"tanh(sin(log(sin(exp(-"+fmt(a,3)+"x^2)))))",[g](ld z){
            ld q=sinl(g(z)); if(q<=1e-14L) return 0.0L; return tanhl(sinl(logl(q)));
        }});
    }
    // Additional nested bounded transforms selected by correlation before the
    // regularized solve. These capture compositions beyond a single unary layer.
    t.push_back({"tanh(cos(log1p(abs(x))))",[](ld z){return tanhl(cosl(log1pl(fabsl(z))));}});
    t.push_back({"sin(cos(x))",[](ld z){return sinl(cosl(z));}});
    t.push_back({"cos(sin(x))",[](ld z){return cosl(sinl(z));}});
    t.push_back({"tanh(sin(x))",[](ld z){return tanhl(sinl(z));}});
    t.push_back({"tanh(cos(x))",[](ld z){return tanhl(cosl(z));}});
    t.push_back({"exp(-abs(x))",[](ld z){return expl(clamp(-fabsl(z),-40.0L,0.0L));}});
    t.push_back({"exp(-x^4)",[](ld z){ld q=z*z;return expl(clamp(-q*q,-40.0L,0.0L));}});

    const vector<ld> expScales = {-2.0L,-1.0L,-0.5L,0.5L,1.0L,2.0L};
    for (ld s : expScales) {
        string label = "exp(" + fmt(s,3) + "*t)";
        t.push_back({label, [tn,s,safeExp](ld z){return safeExp(s*tn(z));}});
    }

    const vector<int> periods = {2,3,4,5,6,8,10,12,16,20,24,32,48,64};
    for (int p : periods) {
        ld w = 2.0L * PI / p;
        t.push_back({"sin(2pi*t/"+to_string(p)+")", [tn,w](ld z){return sinl(w*tn(z));}});
        t.push_back({"cos(2pi*t/"+to_string(p)+")", [tn,w](ld z){return cosl(w*tn(z));}});
    }

    // Nested / composite transforms. These deliberately cover the difficult
    // combinations users commonly encounter without exploding into arbitrary
    // expression trees.
    for (int p : {4,6,8,12,16,24}) {
        ld w = 2.0L * PI / p;
        t.push_back({"exp(sin("+to_string(p)+"-period))",
            [tn,w,safeExp](ld z){return safeExp(sinl(w*tn(z)));}});
        t.push_back({"exp(cos("+to_string(p)+"-period))",
            [tn,w,safeExp](ld z){return safeExp(cosl(w*tn(z)));}});
        t.push_back({"sin(log1p(u))",
            [safeLog](ld z){return sinl(safeLog(z));}});
        t.push_back({"cos(log1p(u))",
            [safeLog](ld z){return cosl(safeLog(z));}});
        t.push_back({"sin(exp(t))",
            [tn,safeExp](ld z){return sinl(safeExp(tn(z)));}});
        t.push_back({"cos(exp(t))",
            [tn,safeExp](ld z){return cosl(safeExp(tn(z)));}});
        break;
    }

    // High-value interaction terms: these are the pieces that make expressions
    // such as log(x) + exp(x)*sin(x) + polynomial trend representable by one model.
    vector<BasisTerm> base = t;
    int original = sizeToInt(t.size());
    for (int i = 0; i < original && (int)t.size() < 54; ++i) {
        if (base[i].isIntercept) continue;
        if (base[i].name.find("exp(") == 0 || base[i].name.find("sin(") == 0 ||
            base[i].name.find("cos(") == 0 || base[i].name.find("log") != string::npos) {
            for (int j = 0; j < original && (int)t.size() < 54; ++j) {
                if (j == i || base[j].isIntercept) continue;
                bool useful = base[j].name.find("sin(") == 0 || base[j].name.find("cos(") == 0 ||
                              base[j].name.find("log") != string::npos || base[j].name.find("exp(") == 0 ||
                              base[j].name == "t" || base[j].name == "t^2";
                if (!useful) continue;
                auto fi = base[i].f, fj = base[j].f;
                string nm = base[i].name + "*" + base[j].name;
                t.push_back({nm, [fi,fj](ld z){return fi(z)*fj(z);}});
            }
        }
    }
    return t;
}

struct BasisFit {
    bool ok = false;
    vector<BasisTerm> terms;
    vector<ld> beta;
    vector<ld> center;
    vector<ld> scale;
    Regularizer reg = Regularizer::Ridge;
    ld lambda = 0;
    function<ld(ld)> predict;
};

BasisFit fitBasis(const vector<ld>& x,
                  const vector<ld>& y,
                  Regularizer reg,
                  ld lambda,
                  int maxFeatures = 20,
                  ld l1Ratio = 0.35L) {
    BasisFit out;
    if (x.size() < 8 || x.size() != y.size()) return out;
    auto all = makeBasisLibrary(x);
    if (all.empty()) return out;

    // Rank non-intercept features by absolute correlation with y, keeping the
    // strongest features before regularization. This is a cheap form of
    // supervised feature selection that avoids huge ill-conditioned systems.
    struct Rank { int idx; ld score; };
    vector<Rank> rank;
    ld ym = meanOf(y);
    ld yss = 0;
    for (ld v : y) yss += (v-ym)*(v-ym);
    for (int j = 1; j < (int)all.size(); ++j) {
        ld fm = 0, num = 0, fss = 0;
        for (ld xx : x) fm += all[j].f(xx);
        fm /= max<size_t>(1, x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            ld f = all[j].f(x[i]) - fm;
            ld q = y[i] - ym;
            num += f*q; fss += f*f;
        }
        ld corr = (fss < 1e-20L || yss < 1e-20L) ? 0 : fabsl(num) / sqrtl(fss*yss);
        if (isfinite(corr)) rank.push_back({j,corr});
    }
    sort(rank.begin(), rank.end(), [](const Rank&a,const Rank&b){return a.score>b.score;});
    maxFeatures = max(4, min(maxFeatures, sizeToInt(rank.size())));

    vector<BasisTerm> selected;
    selected.push_back(all[0]);
    for (int i=0;i<maxFeatures;++i) selected.push_back(all[rank[i].idx]);

    // Standardize non-intercept features; this materially improves ridge/lasso
    // conditioning when log/exp/Fourier/polynomial features coexist.
    const int p = sizeToInt(selected.size());
    out.center.assign(p,0);
    out.scale.assign(p,1);
    for (int j=1;j<p;++j) {
        for (ld xx:x) out.center[j] += selected[j].f(xx);
        out.center[j]/=(ld)x.size();
        ld ss=0;
        for(ld xx:x){ld q=selected[j].f(xx)-out.center[j];ss+=q*q;}
        out.scale[j]=sqrtl(max<ld>(ss/(ld)max<size_t>(1,x.size()-1),1e-18L));
    }

    vector<vector<ld>> A(x.size(), vector<ld>(p));
    for (size_t i=0;i<x.size();++i) {
        A[i][0]=1;
        for (int j=1;j<p;++j) {
            ld v=selected[j].f(x[i]);
            A[i][j]=(v-out.center[j])/out.scale[j];
            if (!isfinite(A[i][j])) A[i][j]=0;
        }
    }

    vector<ld> beta;
    if (!fitRegularizedMatrix(A,y,reg,lambda,beta,l1Ratio)) return out;
    if ((int)beta.size()!=p) return out;

    out.ok=true; out.terms=selected; out.beta=beta; out.reg=reg; out.lambda=lambda;
    auto termsCopy=selected;
    auto center=out.center;
    auto scale=out.scale;
    auto b=beta;
    out.predict=[termsCopy,center,scale,b](ld z){
        if (termsCopy.empty()) return numeric_limits<ld>::quiet_NaN();
        ld ans=b[0];
        for(size_t j=1;j<termsCopy.size();++j){
            ld q=(termsCopy[j].f(z)-center[j])/scale[j];
            if(!isfinite(q)) q=0;
            ans += b[j]*q;
        }
        return ans;
    };
    return out;
}

Model fitBestSingleFeatureModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<8||x.size()!=y.size())return{};
    auto all=makeBasisLibrary(x);if(all.size()<2)return{};
    ld best=numeric_limits<ld>::infinity();int bestIdx=-1;vector<ld>bestC;
    for(int j=1;j<(int)all.size();++j){
        vector<ld>f(x.size());bool ok=true;for(size_t i=0;i<x.size();++i){f[i]=all[j].f(x[i]);if(!isfinite(f[i])){ok=false;break;}}
        if(!ok)continue;
        auto c=linearFit(f,y);if(c.size()!=2)continue;
        ld sse=0;for(size_t i=0;i<x.size();++i){ld p=c[0]+c[1]*f[i];ld e=y[i]-p;sse+=e*e;}
        if(sse<best){best=sse;bestIdx=j;bestC=c;}
    }
    if(bestIdx<0||bestC.size()!=2)return{};
    auto term=all[bestIdx];
    Model m;m.valid=true;m.name="Single nonlinear transformed feature";m.category="Targeted transform identification";m.complexity=2;
    m.predict=[f=term.f,a=bestC[0],b=bestC[1]](ld z){ld q=f(z);return isfinite(q)?a+b*q:numeric_limits<ld>::quiet_NaN();};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(bestC[0],6)+" + ("+fmt(bestC[1],6)+") * "+term.name;
    m.note="Each analytic feature is tested independently before regularized multi-feature mixing; useful when one supplied transform nearly explains the response exactly.";return m;
}

Model makeBasisModel(const vector<ld>& x,const vector<ld>& y,Regularizer reg,ld lambda,int maxFeatures=28) {
    auto f=fitBasis(x,y,reg,lambda,maxFeatures);
    if(!f.ok) return {};
    Model m;
    m.valid=true;
    m.name=regularizerName(reg)+" nonlinear feature-basis model";
    m.category="Machine-learning regression / feature engineering";
    m.complexity=(int)f.terms.size();
    m.predict=f.predict;
    m.nextValue=m.predict(x.back()+1);
    m.fitted.reserve(x.size());
    for(ld xx:x)m.fitted.push_back(m.predict(xx));
    ostringstream os;
    os<<regularizerName(reg)<<" basis, lambda="<<fmt(lambda,4)<<", selected terms="<<f.terms.size()<<": ";
    for(size_t i=0;i<f.terms.size() && i<12;++i){if(i)os<<" + ";os<<f.terms[i].name;}
    if(f.terms.size()>12)os<<" + ...";
    m.formula=os.str();
    m.note="Supervised feature selection + standardized nonlinear basis features + regularization.";
    return m;
}

struct PolyFit {
    bool ok=false;
    vector<ld> c;
    ld xmin=0,xmax=1;
    int degree=0;
    function<ld(ld)> predict;
};

PolyFit fitPolynomial(const vector<ld>& x,const vector<ld>& y,int degree){
    PolyFit out;
    if(x.size()<2||x.size()!=y.size()||degree<0||degree+1>(int)x.size())return out;
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end());
    if(fabsl(xmax-xmin)<1e-18L)return out;
    int p=degree+1;
    vector<vector<ld>>A(x.size(),vector<ld>(p));
    ld mid=(xmin+xmax)/2,half=(xmax-xmin)/2;
    for(size_t i=0;i<x.size();++i){
        ld t=(x[i]-mid)/half,q=1;
        for(int j=0;j<p;++j){A[i][j]=q;q*=t;}
    }
    vector<ld>c;if(!leastSquaresQR(A,y,c))return out;
    out.ok=true;out.c=c;out.xmin=xmin;out.xmax=xmax;out.degree=degree;
    out.predict=[c,mid,half](ld z){
        ld t=(z-mid)/half,val=0,q=1;
        for(ld cj:c){val+=cj*q;q*=t;}
        return val;
    };
    return out;
}

Model fitLinearModel(const vector<ld>& x, const vector<ld>& y) {
    if (x.size() < 3) return {};
    auto c = linearFit(x,y); if(c.size()!=2)return {};
    Model m; m.valid=true; m.name="Linear regression"; m.category="Polynomial / regression"; m.complexity=2;
    m.predict=[a=c[0],b=c[1]](ld z){return a+b*z;};
    m.formula="y = "+fmt(c[0])+" + ("+fmt(c[1])+")x";
    m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.nextValue=m.predict(x.back()+1);
    return m;
}

Model fitPolyModel(const vector<ld>& x, const vector<ld>& y, int degree) {
    auto f=fitPolynomial(x,y,degree);if(!f.ok)return {};
    Model m;m.valid=true;m.name="Polynomial degree "+to_string(degree);m.category="Polynomial regression";m.complexity=degree+1;
    m.predict=f.predict;m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    ostringstream os;os<<"y = ";for(int i=0;i<(int)f.c.size();++i){if(i)os<<" + ";os<<fmt(f.c[i],5)<<" t^"<<i;}
    ld mid=(f.xmin+f.xmax)/2,half=(f.xmax-f.xmin)/2;
    os<<", t=(x-"<<fmt(mid,5)<<")/"<<fmt(half,5);m.formula=os.str();
    if(degree<=14){ for(ld rt:polynomialRealRoots(f.c)){ld rx=mid+half*rt;if(isfinite(rx))m.approxRealRoots.push_back(rx);} }
    return m;
}

// Same underlying polynomial family, but fit in the Chebyshev basis
// (T_0..T_degree evaluated on t in [-1,1]) instead of raw monomials t^k.
// The two bases can represent exactly the same curves, but the Chebyshev
// basis is vastly better conditioned at higher degree -- the monomial fit
// above degrades and effectively caps out well before this one does, so
// this family is what lets genuinely higher-order polynomial shapes stay
// numerically trustworthy. Roots are reported by converting the fitted
// Chebyshev coefficients back to a monomial polynomial and root-finding
// that (a standard, numerically safe conversion since it happens only
// once per fit, not inside the fitting loop itself).
Model fitChebyshevModel(const vector<ld>& x, const vector<ld>& y, int degree) {
    if (x.size() < 3 || x.size() != y.size() || degree < 1 || degree + 1 > (int)x.size()) return {};
    ld xmin=*min_element(x.begin(),x.end()), xmax=*max_element(x.begin(),x.end());
    if (fabsl(xmax-xmin) < 1e-18L) return {};
    ld mid=(xmin+xmax)/2, half=(xmax-xmin)/2;
    int p = degree+1;
    vector<vector<ld>> A(x.size(), vector<ld>(p));
    for (size_t i=0;i<x.size();++i) {
        ld t=(x[i]-mid)/half;
        A[i][0]=1; if(p>1) A[i][1]=t;
        for (int k=2;k<p;++k) A[i][k]=2*t*A[i][k-1]-A[i][k-2];
    }
    vector<ld> c; if (!leastSquaresQR(A,y,c)) return {};
    Model m; m.valid=true; m.name="Chebyshev polynomial degree "+to_string(degree);
    m.category="Numerically-stable high-order polynomial regression"; m.complexity=degree+1;
    m.predict=[c,mid,half](ld z){
        ld t=(z-mid)/half, T0=1, T1=t, s=c[0]*T0; if((int)c.size()>1) s+=c[1]*T1;
        for (int k=2;k<(int)c.size();++k){ ld T2=2*t*T1-T0; s+=c[k]*T2; T0=T1; T1=T2; }
        return s;
    };
    m.nextValue = m.predict(x.back()+1);
    m.fitted.reserve(x.size()); for(ld xx:x) m.fitted.push_back(m.predict(xx));
    ostringstream os; os<<"y = sum_{k=0}^{"<<degree<<"} c_k*T_k(t), t=(x-"<<fmt(mid,5)<<")/"<<fmt(half,5)<<" [Chebyshev basis]";
    m.formula=os.str();
    if (degree<=20) {
        vector<vector<ld>> Tpoly(p);
        Tpoly[0]={1}; if(p>1) Tpoly[1]={0,1};
        for (int k=2;k<p;++k) {
            vector<ld> shifted(Tpoly[k-1].size()+1,0);
            for (size_t i=0;i<Tpoly[k-1].size();++i) shifted[i+1]=2*Tpoly[k-1][i];
            vector<ld> res=shifted;
            for (size_t i=0;i<Tpoly[k-2].size();++i){ if(i>=res.size()) res.resize(i+1,0); res[i]-=Tpoly[k-2][i]; }
            Tpoly[k]=res;
        }
        vector<ld> monomial(p,0);
        for (int k=0;k<p;++k) for (size_t i=0;i<Tpoly[k].size();++i) monomial[i]+=c[k]*Tpoly[k][i];
        for (ld rt: polynomialRealRoots(monomial)) { ld rx=mid+half*rt; if(isfinite(rx)) m.approxRealRoots.push_back(rx); }
    }
    return m;
}

// Newton's divided-difference formula (the general, non-uniform-spacing form
// of a Taylor/finite-difference extrapolation): fits the unique degree-K
// interpolating polynomial through the last K+1 points and extrapolates it
// forward. Unlike the exact finite-difference detector elsewhere in this
// file (which requires perfectly constant K-th differences), this one is an
// *approximate* family -- it is just another candidate scored like anything
// else by rolling-origin validation, so it is only kept when it actually
// forecasts well on data it did not see.
Model fitNewtonTaylorModel(const vector<ld>& x, const vector<ld>& y, int order) {
    const auto n = sizeToInt(y.size());
    if (order < 1 || n < order + 2) return {};
    int start = n - (order + 1);
    vector<ld> xs(x.begin()+start, x.end()), ys(y.begin()+start, y.end());
    int k = (int)xs.size();
    vector<vector<ld>> dd(k, vector<ld>(k, 0));
    for (int i=0;i<k;++i) dd[i][0]=ys[i];
    for (int j=1;j<k;++j) for (int i=0;i+j<k;++i) {
        ld denom = xs[i+j]-xs[i];
        if (fabsl(denom) < 1e-12L) return {};
        dd[i][j] = (dd[i+1][j-1]-dd[i][j-1]) / denom;
    }
    vector<ld> coeffs(k); for (int j=0;j<k;++j) coeffs[j]=dd[0][j];
    Model m; m.valid=true; m.name="Newton divided-difference extrapolation (order "+to_string(order)+")";
    m.category="Local Taylor/interpolation extrapolation"; m.complexity=order+1;
    m.predict=[xs,coeffs,k](ld z){
        ld s=coeffs[k-1];
        for (int j=k-2;j>=0;--j) s = s*(z-xs[j]) + coeffs[j];
        return s;
    };
    m.nextValue = m.predict(x.back()+1);
    m.fitted.reserve(x.size()); for(ld xx:x) m.fitted.push_back(m.predict(xx));
    m.formula = "Newton divided-difference polynomial through the last "+to_string(k)+" points (order "+to_string(order)+")";
    return m;
}


Model fitExpModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<5||any_of(y.begin(),y.end(),[](ld v){return v<=0;}))return{};
    vector<ld>ly(y.size());for(size_t i=0;i<y.size();++i)ly[i]=logl(y[i]);
    auto c=linearFit(x,ly);if(c.size()!=2)return{};
    ld a=expl(c[0]),b=c[1];if(!isfinite(a)||!isfinite(b))return{};
    Model m;m.valid=true;m.name="Exponential";m.category="Nonlinear transformed regression";m.complexity=2;
    m.predict=[a,b](ld z){ld e=b*z;if(e>10000)return numeric_limits<ld>::infinity();return a*expl(e);};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(a)+" exp("+fmt(b)+" x)";return m;
}

Model fitLogModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<5||any_of(x.begin(),x.end(),[](ld v){return v<=0;}))return{};
    vector<ld>lx(x.size());for(size_t i=0;i<x.size();++i)lx[i]=logl(x[i]);
    auto c=linearFit(lx,y);if(c.size()!=2)return{};
    Model m;m.valid=true;m.name="Logarithmic";m.category="Nonlinear transformed regression";m.complexity=2;
    m.predict=[a=c[0],b=c[1]](ld z){return z>0?a+b*logl(z):numeric_limits<ld>::quiet_NaN();};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(c[0])+" + ("+fmt(c[1])+") ln(x)";return m;
}

Model fitPowerModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<5||any_of(x.begin(),x.end(),[](ld v){return v<=0;})||any_of(y.begin(),y.end(),[](ld v){return v<=0;}))return{};
    vector<ld>lx(x.size()),ly(y.size());for(size_t i=0;i<x.size();++i){lx[i]=logl(x[i]);ly[i]=logl(y[i]);}
    auto c=linearFit(lx,ly);if(c.size()!=2)return{};
    ld a=expl(c[0]),b=c[1];
    Model m;m.valid=true;m.name="Power law";m.category="Nonlinear transformed regression";m.complexity=2;
    m.predict=[a,b](ld z){return z>0?a*powl(z,b):numeric_limits<ld>::quiet_NaN();};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(a)+" x^"+fmt(b);return m;
}

Model fitRationalModel(const vector<ld>& x,const vector<ld>& y,int degree=1){
    if(x.size()<10||x.size()!=y.size())return{};
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end()),span=max<ld>(xmax-xmin,1e-9L),mid=(xmin+xmax)/2;
    auto t=[mid,span](ld z){return (z-mid)/(span/2);};
    int d=degree==2?2:1;
    vector<vector<ld>>A(x.size(),vector<ld>(2*d+1));
    for(size_t i=0;i<x.size();++i){
        ld q=t(x[i]),q2=q*q;
        if(d==1) A[i]={1,q,-y[i]*q};
        else A[i]={1,q,q2,-y[i]*q,-y[i]*q2};
    }
    vector<ld>c;if(!leastSquaresQR(A,y,c))return{};
    Model m;m.valid=true;m.name="Rational function regression";m.category="Rational / nonlinear regression";m.complexity=(int)c.size();
    m.predict=[c,t,d](ld z){
        ld q=t(z),num=0,den=1;
        if(d==1){num=c[0]+c[1]*q;den=1+c[2]*q;}
        else {num=c[0]+c[1]*q+c[2]*q*q;den=1+c[3]*q+c[4]*q*q;}
        if(fabsl(den)<1e-10L)return numeric_limits<ld>::quiet_NaN();
        return num/den;
    };
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula=d==1?"(a+b t)/(1+c t)":"(a+b t+c t^2)/(1+d t+e t^2)";
    return m;
}

Model fitSinusoidModel(const vector<ld>& x,const vector<ld>& y,ld w){
    if(x.size()<8||w<=0)return{};
    vector<vector<ld>>A(x.size(),vector<ld>(4));
    for(size_t i=0;i<x.size();++i){A[i][0]=1;A[i][1]=x[i];A[i][2]=sinl(w*x[i]);A[i][3]=cosl(w*x[i]);}
    vector<ld>c;if(!leastSquaresQR(A,y,c))return{};
    Model m;m.valid=true;m.name="Trend + sinusoid";m.category="Trigonometric regression";m.complexity=4;
    m.predict=[c,w](ld z){return c[0]+c[1]*z+c[2]*sinl(w*z)+c[3]*cosl(w*z);};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(c[0])+" + "+fmt(c[1])+"x + "+fmt(c[2])+"sin("+fmt(w)+"x) + "+fmt(c[3])+"cos("+fmt(w)+"x), period="+fmt(2*PI/w);
    return m;
}

Model fitFourierModel(const vector<ld>& x,const vector<ld>& y,ld period,int harmonics){
    if(x.size()<(size_t)max(10,2*harmonics+4)||period<=0||harmonics<1)return{};
    vector<vector<ld>>A(x.size(),vector<ld>(2+2*harmonics));
    for(size_t i=0;i<x.size();++i){
        A[i][0]=1;A[i][1]=x[i];
        for(int k=1;k<=harmonics;++k){
            ld w=2*PI*k/period;A[i][2*k]=sinl(w*x[i]);A[i][2*k+1]=cosl(w*x[i]);
        }
    }
    vector<ld>c;if(!leastSquaresQR(A,y,c))return{};
    Model m;m.valid=true;m.name="Multi-harmonic Fourier regression";m.category="Spectral / Fourier regression";m.complexity=2+2*harmonics;
    m.predict=[c,period,harmonics](ld z){
        ld v=c[0]+c[1]*z;
        for(int k=1;k<=harmonics;++k){ld w=2*PI*k/period;v+=c[2*k]*sinl(w*z)+c[2*k+1]*cosl(w*z);}
        return v;
    };
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="trend + "+to_string(harmonics)+" Fourier harmonics, period="+fmt(period,5);return m;
}

Model fitRecurrenceRidge(const vector<ld>& x,const vector<ld>& y,int order,ld lambda,bool differenced=false){
    (void)x;
    if(y.size()<static_cast<size_t>(order+6)||order<1)return{};
    vector<ld>source=y;
    if(differenced){
        source.resize(y.size()-1);
        for(size_t i=1;i<y.size();++i)source[i-1]=y[i]-y[i-1];
    }
    if(source.size()<=static_cast<size_t>(order))return{};
    vector<vector<ld>>A;vector<ld>b;
    for(size_t i=order;i<source.size();++i){
        vector<ld>row(order+1,1.0L);
        for(int j=1;j<=order;++j)row[j]=source[i-j];
        A.push_back(row);b.push_back(source[i]);
    }
    vector<ld>c;if(!ridgeSolve(A,b,lambda,c)||c.size()!=static_cast<size_t>(order+1))return{};
    Model m;m.valid=true;
    m.name=differenced?"AR("+to_string(order)+") on differences":"Autoregressive AR("+to_string(order)+")";
    m.category=differenced?"Differenced autoregression":"Time-series autoregression";
    m.complexity=order+2;
    const ld xLast = x.empty() ? (ld)y.size() : x.back();
    m.predict=[c, y, order, differenced, xLast](ld z){
        int steps=(int)llroundl(z-xLast);
        if(steps<=0)return numeric_limits<ld>::quiet_NaN();
        vector<ld> work;
        if(differenced){
            work.resize(y.size()-1);
            for(size_t i=1;i<y.size();++i)work[i-1]=y[i]-y[i-1];
        }else work=y;
        ld level=y.back();
        for(int s=0;s<steps;++s){
            ld pred=c[0];
            for(int j=1;j<=order;++j){
                if((int)work.size()<j)return numeric_limits<ld>::quiet_NaN();
                pred+=c[j]*work[work.size()-j];
            }
            work.push_back(pred);
            if(differenced)level+=pred;
        }
        return differenced?level:work.back();
    };
    m.nextValue=m.predict(xLast+1);
    m.fitted=y;
    m.formula=differenced?"Delta(y)_t = c0 + sum(ci Delta(y)_{t-i})":"y_t = c0 + sum(ci y_{t-i})";
    return m;
}

Model fitPronyZModel(const vector<ld>& x,const vector<ld>& y,int order){
    if(x.size()!=y.size()||order<2||y.size()<static_cast<size_t>(order+8)||!isUnitSpacedGrid(x))return{};
    vector<vector<ld>>A;vector<ld>b;
    for(size_t i=order;i<y.size();++i){vector<ld>row(order+1,1.0L);for(int j=1;j<=order;++j)row[j]=y[i-j];A.push_back(row);b.push_back(y[i]);}
    vector<ld>c;if(!leastSquaresQR(A,b,c))return{};
    Model m;m.valid=true;m.name="Prony / Z-transform pole recurrence (order "+to_string(order)+")";m.category="Signal processing / Z-transform system identification";m.complexity=order+2;
    const ld xLast = x.back();
    m.predict=[c,y,order,xLast](ld z){int steps=(int)llroundl(z-xLast);if(steps<=0)return numeric_limits<ld>::quiet_NaN();vector<ld>w=y;for(int s=0;s<steps;++s){ld p=c[0];for(int j=1;j<=order;++j)p+=c[j]*w[w.size()-j];if(!isfinite(p))return numeric_limits<ld>::quiet_NaN();w.push_back(p);}return w.back();};
    m.nextValue=m.predict(xLast+1);m.fitted=y;
    vector<ld>charpoly(order+1,0.0L);charpoly[order]=1.0L;for(int j=1;j<=order;++j)charpoly[order-j]=-c[j];
    auto roots=polynomialRealRoots(charpoly);m.approxRealRoots=roots;
    ostringstream os;os<<"Z-domain recurrence: y_n=c0+sum(c_i y_{n-i}), characteristic roots/poles=";for(size_t i=0;i<roots.size();++i){if(i)os<<",";os<<fmt(roots[i],6);}m.formula=os.str();
    m.note="Prony-style linear recurrence identification; the characteristic polynomial roots describe real Z-plane poles, while complex poles are retained implicitly in the fitted real recurrence.";return m;
}

Model fitHoltModel(const vector<ld>& x,const vector<ld>& y,bool damped){
    if(y.size()<8||x.size()!=y.size())return{};
    ld bestSSE=numeric_limits<ld>::infinity(),bestA=.5L,bestB=.1L,bestPhi=.95L;
    for(int ia=1;ia<=9;++ia){
        ld alpha=ia/10.0L;
        for(int ib=1;ib<=9;++ib){
            ld beta=ib/10.0L;
            vector<ld> level(y.size()),trend(y.size());
            level[0]=y[0];trend[0]=y[1]-y[0];
            ld sse=0;
            for(size_t i=1;i<y.size();++i){
                ld phi=damped?.80L+(ib/10.0L)*.02L:1.0L;
                ld prevLevel=level[i-1],prevTrend=trend[i-1]*phi;
                level[i]=alpha*y[i]+(1-alpha)*(prevLevel+prevTrend);
                trend[i]=beta*(level[i]-prevLevel)+(1-beta)*prevTrend;
                ld p=prevLevel+prevTrend;
                sse+=(y[i]-p)*(y[i]-p);
            }
            if(sse<bestSSE){bestSSE=sse;bestA=alpha;bestB=beta;}
        }
    }
    // A second grid on damping, when requested.
    if(damped){
        for(int ip=80;ip<=99;++ip){
            ld phi=ip/100.0L,alpha=.5L,beta=.2L;
            vector<ld>level(y.size()),trend(y.size());
            level[0]=y[0];trend[0]=y[1]-y[0];ld sse=0;
            for(size_t i=1;i<y.size();++i){
                ld prevLevel=level[i-1],prevTrend=phi*trend[i-1];
                level[i]=alpha*y[i]+(1-alpha)*(prevLevel+prevTrend);
                trend[i]=beta*(level[i]-prevLevel)+(1-beta)*prevTrend;
                sse+=(y[i]-(prevLevel+prevTrend))*(y[i]-(prevLevel+prevTrend));
            }
            if(sse<bestSSE) {bestSSE=sse;bestA=alpha;bestB=beta;bestPhi=phi;}
        }
    }
    ld level=y[0],trend=y[1]-y[0];
    for(size_t i=1;i<y.size();++i){
        ld prevLevel=level,prevTrend=damped?bestPhi*trend:trend;
        level=bestA*y[i]+(1-bestA)*(prevLevel+prevTrend);
        trend=bestB*(level-prevLevel)+(1-bestB)*prevTrend;
    }
    Model m;m.valid=true;m.name=damped?"Damped Holt trend":"Holt double exponential smoothing";
    m.category="Exponential smoothing / state-space-style model";m.complexity=3;
    m.predict=[level,trend,bestPhi,damped,lastX=x.back()](ld z){
        int steps=(int)llroundl(z-lastX);if(steps<1)return numeric_limits<ld>::quiet_NaN();
        if(!damped)return level+steps*trend;
        ld sum=0,p=1;
        for(int i=1;i<=steps;++i){p*=bestPhi;sum+=p;}
        return level+trend*sum;
    };
    m.nextValue=m.predict(x.back()+1);m.fitted=y;
    m.formula=damped?"Holt damped trend state update":"Holt level + trend exponential smoothing";
    return m;
}

Model fitKernelRidge(const vector<ld>& x,const vector<ld>& y,ld sigma,ld lambda){
    if(x.size()<10||sigma<=0||lambda<=0)return{};
    const int n=sizeToInt(x.size());
    vector<vector<ld>>K(n,vector<ld>(n));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){
        ld d=x[i]-x[j];K[i][j]=expl(-(d*d)/(2*sigma*sigma));
    }
    vector<ld>alpha;if(!solveSquareSystem([&]{
        auto A=K;for(int i=0;i<n;++i)A[i][i]+=lambda;return A;
    }(),y,alpha))return{};
    Model m;m.valid=true;m.name="RBF kernel ridge / GP mean surrogate";m.category="Kernel machine learning";
    m.complexity=min(n,50);
    m.predict=[x,alpha,sigma](ld z){ld s=0;for(size_t i=0;i<x.size();++i){ld d=z-x[i];s+=alpha[i]*expl(-(d*d)/(2*sigma*sigma));}return s;};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="RBF kernel K=exp(-(x-x')^2/(2 sigma^2)), lambda="+fmt(lambda,4)+", sigma="+fmt(sigma,4);
    m.note="Kernel ridge is used as a Gaussian-process-style mean function; uncertainty intervals are not claimed.";
    return m;
}

Model fitMeanModel(const vector<ld>& x,const vector<ld>& y){
    if(y.empty())return{};
    ld m=meanOf(y);Model q;q.valid=true;q.name="Mean baseline";q.category="Statistical baseline";q.complexity=1;
    q.predict=[m](ld){return m;};q.nextValue=m;q.fitted.assign(x.size(),m);q.formula="y = sample mean = "+fmt(m);return q;
}

vector<int> candidateDegrees(int n){
    int maxD=min(n-2,8);if(maxD<1)return{};
    vector<int>d;for(int i=1;i<=maxD;++i)d.push_back(i);
    return d;
}

vector<int> candidateWindows(int n){
    vector<int>w;
    auto add=[&](int z){if(z>=8&&z<=n)w.push_back(z);};
    add(n);
    add((int)llround(.80L*n));
    add((int)llround(.60L*n));
    add((int)llround(.40L*n));
    add(max(8,n/4));
    if(n<=64){
        for(int z:{8,12,16,24}) add(z);
    }else if(n<=120){
        for(int z:{16,24,32}) add(z);
    }
    sort(w.begin(),w.end());w.erase(unique(w.begin(),w.end()),w.end());return w;
}

vector<int> candidateHorizons(int n){
    int longH=min(6,max(2,n/12));
    vector<int>h={1,longH};
    h.erase(remove_if(h.begin(),h.end(),[n](int z){return z<1||z>=max(2,n/3+1);}),h.end());
    sort(h.begin(),h.end());h.erase(unique(h.begin(),h.end()),h.end());return h;
}

Model fitSymbolicFeatureSearch(const vector<ld>&,const vector<ld>&,ld);
Model fitLogisticModel(const vector<ld>&,const vector<ld>&,bool);
Model fitPiecewiseLinearModel(const vector<ld>&,const vector<ld>&);
Model fitPiecewiseConstantModel(const vector<ld>&,const vector<ld>&);
Model fitSegmentedLinearModel(const vector<ld>&,const vector<ld>&,int);
Model fitChirpModel(const vector<ld>&,const vector<ld>&,ld,ld);
Model fitDampedSinusoidModel(const vector<ld>&,const vector<ld>&,ld,ld);
Model fitSeasonalTemplateModel(const vector<ld>&,const vector<ld>&,int,int=1);
Model fitPeriodicDifferenceModel(const vector<ld>&,const vector<ld>&,int);
Model fitPiecewiseAdaptiveModel(const vector<ld>&,const vector<ld>&,int);
Model fitOscillatoryEnvelopeModel(const vector<ld>&,const vector<ld>&,ld);
Model fitAdaptiveChirpEnvelopeModel(const vector<ld>&,const vector<ld>&);
Model fitRobustStaircaseProgression(const vector<ld>&,const vector<ld>&);
vector<int> detectChangePoints(const vector<ld>& y,int minSegment=5,ld threshold=3.0L);

Model fitByFamily(const string& family,const vector<ld>& x,const vector<ld>& y,int degree=1,ld param=0,ld param2=0){
    if(family=="mean")return fitMeanModel(x,y);
    if(family=="linear")return fitLinearModel(x,y);
    if(family=="poly")return fitPolyModel(x,y,degree);
    if(family=="chebyshev")return fitChebyshevModel(x,y,degree);
    if(family=="taylor")return fitNewtonTaylorModel(x,y,degree);
    if(family=="exp")return fitExpModel(x,y);
    if(family=="log")return fitLogModel(x,y);
    if(family=="power")return fitPowerModel(x,y);
    if(family=="rational")return fitRationalModel(x,y,degree);
    if(family=="sin")return fitSinusoidModel(x,y,param);
    if(family=="fourier")return fitFourierModel(x,y,param,max(1,degree));
    if(family=="ar")return fitRecurrenceRidge(x,y,degree,max<ld>(1e-8L,param),false);
    if(family=="differenced_ar")return fitRecurrenceRidge(x,y,degree,max<ld>(1e-8L,param),true);
    if(family=="prony")return fitPronyZModel(x,y,degree);
    if(family=="holt")return fitHoltModel(x,y,false);
    if(family=="damped_holt")return fitHoltModel(x,y,true);
    if(family=="piecewise")return fitPiecewiseLinearModel(x,y);
    if(family=="piecewise_const")return fitPiecewiseConstantModel(x,y);
    if(family=="segmented")return fitSegmentedLinearModel(x,y,degree);
    if(family=="piecewise_adaptive")return fitPiecewiseAdaptiveModel(x,y,degree);
    if(family=="osc_envelope")return fitOscillatoryEnvelopeModel(x,y,param);
    if(family=="adaptive_chirp_envelope")return fitAdaptiveChirpEnvelopeModel(x,y);
    if(family=="robust_stair")return fitRobustStaircaseProgression(x,y);
    if(family=="seasonal")return fitSeasonalTemplateModel(x,y,(int)llround(param),degree);
    if(family=="periodic_delta")return fitPeriodicDifferenceModel(x,y,(int)llround(param));
    if(family=="chirp")return fitChirpModel(x,y,param,param2);
    if(family=="damped_sine")return fitDampedSinusoidModel(x,y,param,param2);
    if(family=="root_poly")return fitRootGuidedPolynomial(x,y,degree,(int)max<ld>(1,param));
    if(family=="logistic")return fitLogisticModel(x,y,false);
    if(family=="gompertz")return fitLogisticModel(x,y,true);
    if(family=="symbolic")return fitSymbolicFeatureSearch(x,y,param);
    if(family=="single_feature")return fitBestSingleFeatureModel(x,y);
    if(family=="rbf")return fitKernelRidge(x,y,param,0.02L);
    if(family=="ridge_basis")return makeBasisModel(x,y,Regularizer::Ridge,param,32);
    if(family=="lasso_basis")return makeBasisModel(x,y,Regularizer::Lasso,param,30);
    if(family=="elastic_basis")return makeBasisModel(x,y,Regularizer::ElasticNet,param,30);
    if(family=="huber_basis")return makeBasisModel(x,y,Regularizer::Huber,param,30);
    return {};
}

bool isUnitSpacedGrid(const vector<ld>& x) {
    if (x.size() < 2) return true;
    for (size_t i=1;i<x.size();++i)
        if (fabsl((x[i]-x[i-1])-1.0L) > 1e-8L) return false;
    return true;
}

// Naive DFT-based period detection: transforms the (mean-removed) series to
// the frequency domain, finds the strongest peaks in the magnitude
// spectrum, and converts them back to candidate periods. This lets the
// periodic-family search (sin/fourier) target whatever period the data
// actually contains instead of only the fixed small-integer periods in the
// hand-picked list below -- the two are merged, not one replacing the
// other. O(n^2), fine for the sequence lengths this tool targets.
vector<int> dftDominantPeriods(const vector<ld>& y, int topK = 3) {
    const auto n = sizeToInt(y.size());
    if (n < 12) return {};
    ld mean = meanOf(y);
    vector<pair<ld,int>> spectrum; // (magnitude, bin)
    int maxBin = n/2;
    for (int k=1;k<=maxBin;++k) {
        ld re=0, im=0, w=2*PI*k/n;
        for (int i=0;i<n;++i) { ld v=y[i]-mean; re += v*cosl(w*i); im -= v*sinl(w*i); }
        spectrum.push_back({hypotl(re,im), k});
    }
    sort(spectrum.begin(), spectrum.end(), [](auto&A,auto&B){return A.first>B.first;});
    vector<int> periods;
    for (int i=0;i<(int)spectrum.size() && (int)periods.size()<topK; ++i) {
        int k = spectrum[i].second;
        if (k<=0) continue;
        int period = (int)llround((ld)n/k);
        if (period>=2 && period<=n && find(periods.begin(),periods.end(),period)==periods.end()) periods.push_back(period);
    }
    return periods;
}

// Residual/autocorrelation period detector.  Unlike a DFT-only approach this
// evaluates candidate integer lags directly, so trend+periodic sequences and
// repeating increment patterns are not forced into a straight-line model.
int bestDiscretePeriod(const vector<ld>& x,const vector<ld>& y,int maxPeriod=-1){
    if(x.size()!=y.size() || y.size()<10 || !isUnitSpacedGrid(x)) return -1;
    const int n=sizeToInt(y.size());
    if(maxPeriod<0) maxPeriod=min(48,n/2);
    maxPeriod=max(2,min(maxPeriod,n/2));
    auto c=linearFit(x,y);
    vector<ld>res(y.size());
    for(size_t i=0;i<y.size();++i)res[i]=y[i]-(c.empty()?meanOf(y):c[0]+c[1]*x[i]);
    const ld scale=max<ld>(robustScale(res),1e-12L);
    ld best=numeric_limits<ld>::infinity(); int bestP=-1;
    for(int p=2;p<=maxPeriod;++p){
        if(n/p<2) continue;
        ld e=0,w=0;
        for(int i=p;i<n;++i){e+=fabsl(res[i]-res[i-p]);w+=1;}
        ld score=(w?e/w:1e100L)/scale;
        if(score<best){best=score;bestP=p;}
    }
    // A period shorter than two cycles is not enough structural evidence.
    return (bestP>0 && best<1.75L) ? bestP : -1;
}

vector<int> detectPeriodicCandidates(const vector<ld>& x,const vector<ld>& y){
    vector<int>out;
    int p=bestDiscretePeriod(x,y); if(p>0) out.push_back(p);
    auto dft=dftDominantPeriods(y,5); out.insert(out.end(),dft.begin(),dft.end());
    if(isUnitSpacedGrid(x) && y.size()>=12){
        vector<ld>d;d.reserve(y.size()-1);for(size_t i=1;i<y.size();++i)d.push_back(y[i]-y[i-1]);
        int pd=bestDiscretePeriod(vector<ld>(d.size(),1.0L),d);if(pd>0)out.push_back(pd);
    }
    sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());
    return out;
}

vector<FamilySpec> makeSpecs(const vector<ld>& x,const vector<ld>& y){
    vector<FamilySpec>s;
    int n=sizeToInt(y.size());
    s.push_back({"mean",1,0});
    s.push_back({"linear",1,0});
    for(int d:candidateDegrees(n))s.push_back({"poly",d,0});
    for(int d:{6,8,10,12,15,18}) if(d+1<=n) s.push_back({"chebyshev",d,0});
    for(int ord:{2,3,4,5,6}) if(n>=ord+2) s.push_back({"taylor",ord,0});
    s.push_back({"exp",1,0});s.push_back({"log",1,0});s.push_back({"power",1,0});
    s.push_back({"rational",1,0});s.push_back({"rational",2,0});
    s.push_back({"holt",1,0});s.push_back({"damped_holt",1,0});
    s.push_back({"piecewise",1,0});
    s.push_back({"piecewise_const",1,0});
    for(int k=1;k<=3;++k) s.push_back({"segmented",k,0});
    for(int d=1;d<=4;++d){
        s.push_back({"root_poly",d,1.0L});
        s.push_back({"root_poly",d,2.0L});
        s.push_back({"root_poly",d,3.0L});
    }
    s.push_back({"logistic",1,0});
    s.push_back({"gompertz",1,0});
    s.push_back({"symbolic",1,0.01L});
    s.push_back({"single_feature",1,0});
    vector<ld> lambdas={1e-4L,1e-2L,1e-1L,1.0L};
    for(ld l:lambdas)s.push_back({"ridge_basis",1,l});
    for(ld l:{1e-3L,1e-2L})s.push_back({"lasso_basis",1,l});
    for(ld l:{1e-3L,1e-2L})s.push_back({"elastic_basis",1,l});
    s.push_back({"huber_basis",1,1e-2L});
    if (isUnitSpacedGrid(x)) {
        for(int p=1;p<=min(5,max(1,n/4));++p){
            for(ld l:{1e-3L,1e-1L}){s.push_back({"ar",p,l});s.push_back({"differenced_ar",p,l});}
        }
        for(int p=2;p<=min(6,max(2,n/5));++p) s.push_back({"prony",p,0});
    }
    if(n>=16){
        const vector<ld> w1={0.0L,0.5L,1.0L,-0.5L,-1.0L};
        const vector<ld> w2={0.0L,0.25L,0.5L,1.0L,2.0L};
        for(ld a:w1) for(ld b:w2) if(fabsl(a)+fabsl(b)>0) s.push_back({"chirp",1,a,b});
    }
    set<int>periods={2,3,4,5,6,8,10,12,15,18,20,24,30,36,48};
    ld extremaPeriod=estimateExtremaPeriod(x,y);
    if(isfinite(extremaPeriod)){int ep=max(3,(int)llround(extremaPeriod));if(ep>=3&&ep<=n/2)periods.insert(ep);}
    for(int p:dftDominantPeriods(y)) periods.insert(p);
    for(int p:detectPeriodicCandidates(x,y)) periods.insert(p);
    for(int p:periods)if(p>=2&&p<=max(2,n))s.push_back({"sin",1,2*PI/p});
    for(int p:periods)if(p>=3&&p<=max(3,n))for(int k: {1,2,3,4})s.push_back({"fourier",k,(ld)p});
    // Seasonal-template and repeating-increment models target cases such as
    // 10,20,11,21,12,22,... where plain trend fitting looks deceptively linear.
    for(int p:detectPeriodicCandidates(x,y)) if(p>=2 && p<=n/2){
        s.push_back({"seasonal",1,(ld)p});
        s.push_back({"seasonal",2,(ld)p});
        if(isUnitSpacedGrid(x)) s.push_back({"periodic_delta",1,(ld)p});
    }
    if(isfinite(extremaPeriod)&&extremaPeriod>=3&&extremaPeriod<=n/2){
        s.push_back({"osc_envelope",1,extremaPeriod});
    }
    if(n>=20){
        s.push_back({"adaptive_chirp_envelope",1,0});
    }
    if(n>=14){
        s.push_back({"piecewise_adaptive",1,0});
        s.push_back({"piecewise_adaptive",2,0});
    }
    // Damped sinusoid / RLC-style transient response: reuse the same candidate
    // ringing periods already derived above (fixed list + DFT peaks), paired with
    // a small decay-rate grid scaled to the observed x-span so both fast and slow
    // decays are covered without an expensive continuous search.
    if(n>=14){
        ld span=x.empty()?1:(*max_element(x.begin(),x.end())-*min_element(x.begin(),x.end()));
        vector<ld> alphas={0.0L,0.5L/max<ld>(span,1e-6L),1.0L/max<ld>(span,1e-6L),2.0L/max<ld>(span,1e-6L),4.0L/max<ld>(span,1e-6L)};
        for(int p:periods) if(p>=2&&p<=max(2,n)) for(ld a:alphas) s.push_back({"damped_sine",1,2*PI/p,a});
    }
    // RBF is most useful for moderately large windows.
    for(ld sigmaMult:{0.5L,1.5L}) {
        ld span=x.empty()?1:(*max_element(x.begin(),x.end())-*min_element(x.begin(),x.end()));
        s.push_back({"rbf",1,max<ld>(span*sigmaMult,1e-3L)});
    }
    return s;
}

ld smapePct(const vector<ld>& y,const vector<ld>& p){
    if(y.empty()||y.size()!=p.size())return numeric_limits<ld>::infinity();
    ld s=0;
    for(size_t i=0;i<y.size();++i)s+=200.0L*fabsl(y[i]-p[i])/max<ld>(fabsl(y[i])+fabsl(p[i]),1e-12L);
    return s/y.size();
}

ld directionalAccuracy(const vector<ld>& actual,const vector<ld>& pred){
    if(actual.size()!=pred.size()||actual.size()<2)return numeric_limits<ld>::quiet_NaN();
    int hit=0,total=0;
    for(size_t i=1;i<actual.size();++i){
        ld da=actual[i]-actual[i-1],dp=pred[i]-pred[i-1];
        if(fabsl(da)<1e-18L)continue;
        if((da>0&&dp>0)||(da<0&&dp<0))++hit;
        ++total;
    }
    return total?((ld)hit/total):numeric_limits<ld>::quiet_NaN();
}

ld correlation(const vector<ld>& a,const vector<ld>& b){
    if(a.size()!=b.size()||a.size()<2)return numeric_limits<ld>::quiet_NaN();
    ld am=meanOf(a),bm=meanOf(b),num=0,aa=0,bb=0;
    for(size_t i=0;i<a.size();++i){ld x=a[i]-am,y=b[i]-bm;num+=x*y;aa+=x*x;bb+=y*y;}
    if(aa<1e-24L||bb<1e-24L)return numeric_limits<ld>::quiet_NaN();
    return num/sqrtl(aa*bb);
}

vector<ld> selectValidationOrigins(int firstOrigin,int lastOrigin,int maxOrigins=30){
    vector<ld>dummy;
    vector<int>o;
    if(lastOrigin<firstOrigin)return dummy;
    int count=lastOrigin-firstOrigin+1;
    int step=max(1,(count+maxOrigins-1)/maxOrigins);
    for(int x=firstOrigin;x<=lastOrigin;x+=step)o.push_back(x);
    if(o.empty()||o.back()!=lastOrigin)o.push_back(lastOrigin);
    vector<ld>out;for(int z:o)out.push_back(z);return out;
}

bool absurdPrediction(const Model& m,const vector<ld>& y,const vector<ld>& preds){
    ld scale=max({robustScale(y),1.0L,fabsl(quantile(y,.5L))});
    for(ld q:preds){
        if(!isfinite(q)||fabsl(q)>1e300L)return true;
        ld ratio=fabsl(q)/scale;
        if(ratio>1e8L&&m.metrics.score<90)return true;
        if(m.complexity>50&&ratio>1e6L)return true;
    }
    return false;
}

bool strictToleranceExceeded(ld deviationPct,ld tolerancePct,bool strict){
    return strict ? deviationPct>tolerancePct+1e-12L:false;
}

ld maxAllowedValidationDeviation(ld tolerancePct,bool strict){
    return strict?tolerancePct:max(30.0L,tolerancePct*3.0L);
}

struct ChunkValidation {
    bool ok=false;
    Metrics metrics;
    vector<ld> residuals;
    string reason;
};

ChunkValidation validatePiecewiseConstantRolling(const vector<ld>&x,const vector<ld>&y,int window,
                                                ld tolerancePct,bool strictTolerance){
    ChunkValidation out;
    const int n=sizeToInt(y.size());
    if(n<10||window<8||window>n-2)return out;
    const ld scale=max({robustScale(y),medianAbs(y)*0.25L,1e-12L});
    vector<ld> horizons; for(int h:candidateHorizons(n))horizons.push_back(h);
    vector<Metrics> blocks; vector<ld>allActual,allPred,allDev;
    int transitionBlocks=0;
    for(ld hh:horizons){
        int h=(int)hh, firstOrigin=window, lastOrigin=n-h;
        if(lastOrigin<firstOrigin)continue;
        auto origins=selectValidationOrigins(firstOrigin,lastOrigin,n<=160?24:48);
        for(ld oo:origins){
            int origin=(int)oo,start=max(0,origin-window);
            vector<ld>tx(x.begin()+start,x.begin()+origin),ty(y.begin()+start,y.begin()+origin);
            Model m=fitPiecewiseConstantModel(tx,ty);
            if(!m.valid||!m.predict)continue;
            vector<ld>actual,pred,dv; actual.reserve(h);pred.reserve(h);dv.reserve(h);
            bool bad=false;
            for(int i=origin;i<origin+h;++i){
                ld p=m.predict(x[i]);
                if(!isfinite(p)){bad=true;break;}
                actual.push_back(y[i]);pred.push_back(p);dv.push_back(relativeDeviationPct(y[i],p,scale));
            }
            if(bad||actual.empty())continue;
            // A zero-order hold cannot infer the time of an unobserved future step.
            // A fold whose future contains a clear level jump is therefore transition-ambiguous,
            // not evidence that persistence inside the current plateau is a bad model.
            vector<ld>trDiff;
            for(size_t i=1;i<ty.size();++i)trDiff.push_back(ty[i]-ty[i-1]);
            ld td=quantile(trDiff,.5L);
            vector<ld>tad;for(ld q:trDiff)tad.push_back(fabsl(q-td));
            ld diffNoise=max<ld>(1.4826L*quantile(tad,.5L),1e-9L);
            ld jumpThreshold=max<ld>(4.0L*diffNoise,2.0L*tolerancePct/100.0L*scale);
            bool transition=false;
            for(size_t i=1;i<actual.size();++i)if(fabsl(actual[i]-actual[i-1])>jumpThreshold){transition=true;break;}
            if(transition){++transitionBlocks;continue;}
            auto [rmse,mae]=errorMetrics(actual,pred);
            Metrics mm; mm.nTrain=origin-start;mm.nValid=(int)actual.size();mm.rmse=rmse;mm.mae=mae;mm.r2=r2Score(actual,pred);
            mm.hitRate=count_if(dv.begin(),dv.end(),[&](ld q){return q<=tolerancePct;})/(ld)dv.size();
            mm.meanDeviationPct=meanOf(dv);mm.medianDeviationPct=quantile(dv,.5L);mm.maxDeviationPct=*max_element(dv.begin(),dv.end());mm.p90DeviationPct=quantile(dv,.9L);
            mm.smapePct=smapePct(actual,pred);mm.directionalAccuracy=directionalAccuracy(actual,pred);mm.correlation=correlation(actual,pred);mm.validationBlocks=1;
            bool pass=strictTolerance?(mm.p90DeviationPct<=tolerancePct&&mm.maxDeviationPct<=1.25L*tolerancePct):(mm.medianDeviationPct<=tolerancePct&&mm.p90DeviationPct<=2.0L*tolerancePct);
            mm.blockPassRate=pass?1.0L:0.0L;blocks.push_back(mm);
            out.residuals.insert(out.residuals.end(),actual.size(),0.0L);
            for(size_t ri=0;ri<actual.size();++ri)out.residuals[out.residuals.size()-actual.size()+ri]=actual[ri]-pred[ri];
            allActual.insert(allActual.end(),actual.begin(),actual.end());allPred.insert(allPred.end(),pred.begin(),pred.end());allDev.insert(allDev.end(),dv.begin(),dv.end());
        }
    }
    if(blocks.size()<3||allActual.empty()){out.reason="insufficient within-plateau validation blocks";return out;}
    int total=0,passed=0;ld wr=0,wm=0,wr2=0,wh=0,wsm=0,wda=0,wcor=0;int daN=0,corN=0;
    for(const auto&b:blocks){int w=max(1,b.nValid);total+=w;wr+=b.rmse*w;wm+=b.mae*w;wr2+=b.r2*w;wh+=b.hitRate*w;wsm+=b.smapePct*w;if(b.blockPassRate>0.5L)++passed;if(isfinite(b.directionalAccuracy)){wda+=b.directionalAccuracy*w;daN+=w;}if(isfinite(b.correlation)){wcor+=b.correlation*w;corN+=w;}}
    Metrics m;m.nTrain=blocks.front().nTrain;m.nValid=total;m.validationBlocks=sizeToInt(blocks.size());m.rmse=wr/total;m.mae=wm/total;m.r2=wr2/total;m.hitRate=wh/total;m.smapePct=wsm/total;
    m.meanDeviationPct=meanOf(allDev);m.medianDeviationPct=quantile(allDev,.5L);m.p90DeviationPct=quantile(allDev,.9L);m.maxDeviationPct=*max_element(allDev.begin(),allDev.end());
    m.directionalAccuracy=daN?wda/daN:numeric_limits<ld>::quiet_NaN();m.correlation=corN?wcor/corN:numeric_limits<ld>::quiet_NaN();
    m.normalizedRMSE=m.rmse/max(robustScale(y),1e-12L);
    ld absErr=0,sumAbs=0,naive=0;for(size_t i=0;i<allActual.size();++i){absErr+=fabsl(allActual[i]-allPred[i]);sumAbs+=fabsl(allActual[i]);}
    for(size_t i=1;i<y.size();++i)naive+=fabsl(y[i]-y[i-1]);
    m.wapePct=100.0L*absErr/max<ld>(sumAbs,1e-12L);m.mase=m.mae/max<ld>(naive/max<size_t>(1,y.size()-1),1e-12L);m.blockPassRate=(ld)passed/blocks.size();m.rollingValidated=true;
    const ld tol=max<ld>(tolerancePct,1e-6L);
    m.stable=strictTolerance?(m.p90DeviationPct<=tol&&m.blockPassRate>=.70L):(m.medianDeviationPct<=tol&&m.p90DeviationPct<=2.0L*tol&&m.blockPassRate>=.60L);
    m.scoreRMSE=1/(1+m.normalizedRMSE);
    m.scoreMAE=1/(1+m.wapePct/100);
    m.scoreP90=expl(-m.p90DeviationPct/tol);
    m.scoreR2=(isfinite(m.r2)&&m.r2>=0)?clamp01(m.r2):1.0L/(1.0L+max<ld>(0,m.mase));
    // Direction and correlation are not applicable to a persistent plateau.
    m.scoreDirection=1.0L;m.scoreCorrelation=1.0L;
    m.scoreStability=m.blockPassRate;m.scoreComplexity=.92L;
    // Use only diagnostics that are meaningful for this specialized model.
    ld product=max<ld>(1e-18L,m.scoreRMSE*m.scoreMAE*m.scoreP90*max<ld>(1e-6L,m.scoreR2)*max<ld>(1e-6L,m.scoreStability)*m.scoreComplexity);
    m.score=100*powl(product,1.0L/6.0L);
    out.reason="within-plateau validation; transition-ambiguous folds excluded (zero-order hold predicts persistence, not transition timing)";
    out.ok=true;out.metrics=m;return out;
}

ChunkValidation validateRolling(const vector<ld>&x,const vector<ld>&y,const FamilySpec&spec,int window,
                                ld tolerancePct,bool strictTolerance){
    ChunkValidation out; int n=sizeToInt(y.size());
    if(n<10||window<8||window>n-2)return out;
    if(spec.family=="piecewise_const") return validatePiecewiseConstantRolling(x,y,window,tolerancePct,strictTolerance);
    const vector<ld> horizons = [&](){vector<ld>h;for(int q:candidateHorizons(n))h.push_back(q);return h;}();
    vector<Metrics> blocks; vector<ld> allActual,allPred,allDev;
    const ld scale=max({robustScale(y),medianAbs(y)*0.25L,1e-12L});
    const int minTrain=max(8,spec.degree+3);
    for(ld hh:horizons){
        int h=(int)hh, firstOrigin=window,lastOrigin=n-h;
        if(lastOrigin<firstOrigin)continue;
        auto origins=selectValidationOrigins(firstOrigin,lastOrigin,n<=160?24:48);
        for(ld oo:origins){
            int origin=(int)oo,start=max(0,origin-window);
            if(origin-start<minTrain)continue;
            vector<ld>tx(x.begin()+start,x.begin()+origin),ty(y.begin()+start,y.begin()+origin);
            Model m=fitByFamily(spec.family,tx,ty,spec.degree,spec.param,spec.param2);
            if(!m.valid||!m.predict)continue;
            vector<ld>actual,pred,dv;actual.reserve(h);pred.reserve(h);dv.reserve(h);
            bool bad=false;
            for(int i=origin;i<origin+h;++i){
                ld p=m.predict(x[i]);
                if(!isfinite(p)||fabsl(p)>1e300L){bad=true;break;}
                actual.push_back(y[i]);pred.push_back(p);dv.push_back(relativeDeviationPct(y[i],p,scale));
            }
            if(bad||actual.empty())continue;
            auto [rmse,mae]=errorMetrics(actual,pred);
            Metrics mm;mm.nTrain=origin-start;mm.nValid=(int)actual.size();
            mm.rmse=rmse;mm.mae=mae;mm.r2=r2Score(actual,pred);
            mm.hitRate=count_if(dv.begin(),dv.end(),[&](ld q){return q<=tolerancePct;})/(ld)dv.size();
            mm.meanDeviationPct=meanOf(dv);mm.medianDeviationPct=quantile(dv,.5L);
            mm.maxDeviationPct=*max_element(dv.begin(),dv.end());mm.p90DeviationPct=quantile(dv,.9L);
            mm.smapePct=smapePct(actual,pred);mm.directionalAccuracy=directionalAccuracy(actual,pred);mm.correlation=correlation(actual,pred);
            bool pass = strictTolerance ? (mm.p90DeviationPct<=tolerancePct && mm.maxDeviationPct<=tolerancePct*1.25L)
                                        : (mm.p90DeviationPct<=2.0L*tolerancePct && mm.medianDeviationPct<=tolerancePct);
            mm.blockPassRate=pass?1.0L:0.0L;blocks.push_back(mm);
            out.residuals.insert(out.residuals.end(),actual.size(),0.0L);
            for(size_t ri=0;ri<actual.size();++ri) out.residuals[out.residuals.size()-actual.size()+ri]=actual[ri]-pred[ri];
            allActual.insert(allActual.end(),actual.begin(),actual.end());allPred.insert(allPred.end(),pred.begin(),pred.end());allDev.insert(allDev.end(),dv.begin(),dv.end());
        }
    }
    if(blocks.empty()||allActual.empty()){out.reason="no usable moving-window validation slices";return out;}
    ld wr=0,wm=0,wr2=0,wh=0,wsm=0,wda=0,wcor=0;int total=0,daN=0,corN=0,passed=0;
    for(const auto&b:blocks){int w=max(1,b.nValid);total+=w;wr+=b.rmse*w;wm+=b.mae*w;wr2+=b.r2*w;wh+=b.hitRate*w;wsm+=b.smapePct*w;if(b.blockPassRate>0.5L)++passed;if(isfinite(b.directionalAccuracy)){wda+=b.directionalAccuracy*w;daN+=w;}if(isfinite(b.correlation)){wcor+=b.correlation*w;corN+=w;}}
    Metrics agg;agg.nTrain=blocks.front().nTrain;agg.nValid=total;agg.validationBlocks=sizeToInt(blocks.size());
    agg.rmse=wr/total;agg.mae=wm/total;agg.r2=wr2/total;agg.hitRate=wh/total;agg.smapePct=wsm/total;
    agg.meanDeviationPct=meanOf(allDev);agg.medianDeviationPct=quantile(allDev,.5L);agg.p90DeviationPct=quantile(allDev,.9L);agg.maxDeviationPct=*max_element(allDev.begin(),allDev.end());
    agg.directionalAccuracy=daN?wda/daN:numeric_limits<ld>::quiet_NaN();agg.correlation=corN?wcor/corN:numeric_limits<ld>::quiet_NaN();
    agg.normalizedRMSE=agg.rmse/scale;
    ld sumAbs=0;for(ld q:allActual)sumAbs+=fabsl(q);
    ld sumNaive=0;for(size_t i=1;i<y.size();++i)sumNaive+=fabsl(y[i]-y[i-1]);
    ld absErr=0;for(size_t i=0;i<allActual.size();++i)absErr+=fabsl(allActual[i]-allPred[i]);
    agg.wapePct=100.0L*absErr/max<ld>(sumAbs,1e-12L);
    agg.mase=agg.mae/max<ld>(sumNaive/max<size_t>(1,y.size()-1),1e-12L);
    agg.blockPassRate=(ld)passed/blocks.size();agg.rollingValidated=true;agg.blockMeanDeviationPct.clear();agg.blockMaxDeviationPct.clear();
    for(const auto&b:blocks){agg.blockMeanDeviationPct.push_back(b.meanDeviationPct);agg.blockMaxDeviationPct.push_back(b.maxDeviationPct);}
    const ld tol=max<ld>(tolerancePct,1e-6L);
    agg.stable = strictTolerance ? (agg.p90DeviationPct<=tol && agg.blockPassRate>=0.70L)
                                 : (agg.medianDeviationPct<=tol && agg.p90DeviationPct<=2.0L*tol && agg.blockPassRate>=0.60L && isfinite(agg.normalizedRMSE));
    agg.scoreRMSE=1.0L/(1.0L+max<ld>(0,agg.normalizedRMSE));
    agg.scoreMAE=1.0L/(1.0L+max<ld>(0,agg.wapePct/100.0L));
    agg.scoreP90=expl(-agg.p90DeviationPct/tol);
    agg.scoreR2=clamp01(agg.r2);
    agg.scoreDirection=isfinite(agg.directionalAccuracy)?agg.directionalAccuracy:0.5L;
    agg.scoreCorrelation=isfinite(agg.correlation)?clamp01((agg.correlation+1.0L)/2.0L):0.5L;
    agg.scoreStability=agg.blockPassRate;
    agg.scoreComplexity=1.0L/sqrtl(1.0L+0.03L*max(1,spec.degree+1));
    ld product=max<ld>(1e-18L,agg.scoreRMSE*agg.scoreMAE*agg.scoreP90*max<ld>(1e-6L,agg.scoreR2)*max<ld>(1e-6L,agg.scoreDirection)*max<ld>(1e-6L,agg.scoreCorrelation)*max<ld>(1e-6L,agg.scoreStability)*agg.scoreComplexity);
    agg.score=100.0L*powl(product,1.0L/8.0L);
    out.ok=true;out.metrics=agg;return out;
}

// --------------------- v5 diagnostics helpers -------------------
vector<int> detectChangePoints(const vector<ld>& y, int minSegment, ld threshold) {
    vector<int> cp;
    int n=sizeToInt(y.size());
    if(n<2*minSegment+1) return cp;
    for(int i=minSegment;i<=n-minSegment;++i){
        vector<ld> left(y.begin()+i-minSegment,y.begin()+i);
        vector<ld> right(y.begin()+i,y.begin()+i+minSegment);
        ld lm=quantile(left,.5L), rm=quantile(right,.5L);
        vector<ld> combined=left; combined.insert(combined.end(),right.begin(),right.end());
        ld sc=max<ld>(1.4826L*quantile([&](){vector<ld>d;ld m=quantile(combined,.5L);for(ld v:combined)d.push_back(fabsl(v-m));return d;}(),.5L),1e-9L);
        ld z=fabsl(rm-lm)/sc;
        if(z>=threshold){
            if(cp.empty() || i-cp.back()>=minSegment){ cp.push_back(i); }
            else if(z>=threshold*1.25L){ cp.back()=i; }
        }
    }
    // Avoid long trains of points caused by a single broad transition.
    vector<int> out;
    for(int c:cp){ if(out.empty() || c-out.back()>=minSegment*2) out.push_back(c); }
    return out;
}

vector<ld> empiricalForecastBand(const vector<ld>& pointPred, const vector<ld>& residuals, ld level=0.90L){
    vector<ld> half(pointPred.size(),numeric_limits<ld>::quiet_NaN());
    vector<ld>a;a.reserve(residuals.size());
    for(ld r:residuals)if(isfinite(r))a.push_back(fabsl(r));
    if(pointPred.empty()||a.size()<6)return half;
    ld q=quantile(a,0.5L+level/2.0L);
    if(!isfinite(q))return half;
    for(size_t h=0;h<pointPred.size();++h)half[h]=q*sqrtl((ld)h+1.0L);
    return half;
}

// A small, bounded symbolic-regression search. It composes a curated set of
// transformations and products, then lets regularization decide which terms
// survive. This gives expression-discovery behavior without unrestricted
// expression-tree explosion.
Model fitSymbolicFeatureSearch(const vector<ld>& x,const vector<ld>& y,ld lambda=0.01L){
    auto m=makeBasisModel(x,y,Regularizer::ElasticNet,lambda,24);
    if(!m.valid)return {};
    m.name="Bounded symbolic regression / sparse expression search";
    m.category="Symbolic regression + regularized ML";
    m.note="Curated expression grammar + feature selection + Elastic Net; not an unrestricted theorem prover.";
    return m;
}

Model fitLogisticModel(const vector<ld>& x,const vector<ld>& y,bool gompertz=false){
    if(x.size()<12||x.size()!=y.size())return{};
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end()),span=max<ld>(xmax-xmin,1e-9L);
    ld best=numeric_limits<ld>::infinity(); vector<ld> bestC; ld bestK=1,bestX0=(xmin+xmax)/2;
    vector<ld> ks={0.5L,1.0L,2.0L,3.5L,5.0L};
    vector<ld> centers={0.15L,0.30L,0.45L,0.50L,0.55L,0.70L,0.85L};
    for(ld k0:ks){
        ld k=k0/span;
        for(ld q:centers){
            ld x0=xmin+q*span;
            vector<vector<ld>>A(x.size(),vector<ld>(gompertz?4:4));
            for(size_t i=0;i<x.size();++i){
                ld u=k*(x[i]-x0);
                ld sig=1.0L/(1.0L+expl(clamp(-u,-30.0L,30.0L)));
                ld g=expl(-expl(clamp(-u,-30.0L,30.0L)));
                A[i][0]=1;A[i][1]=x[i];A[i][2]=gompertz?g:sig;A[i][3]=x[i]*(gompertz?g:sig);
            }
            vector<ld>c;if(!leastSquaresQR(A,y,c))continue;
            ld sse=0;for(size_t i=0;i<x.size();++i){ld p=0;for(int j=0;j<4;++j)p+=A[i][j]*c[j];ld e=y[i]-p;sse+=e*e;}
            if(sse<best){best=sse;bestC=c;bestK=k;bestX0=x0;}
        }
    }
    if(bestC.empty())return{};
    Model m;m.valid=true;m.name=gompertz?"Gompertz + trend":"Logistic sigmoid + trend";m.category="Nonlinear sigmoid regression";m.complexity=4;
    m.predict=[c=bestC,k=bestK,x0=bestX0,gompertz](ld z){
        ld u=k*(z-x0),base=gompertz?expl(-expl(clamp(-u,-30.0L,30.0L))):1.0L/(1.0L+expl(clamp(-u,-30.0L,30.0L)));
        return c[0]+c[1]*z+c[2]*base+c[3]*z*base;
    };
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula=(gompertz?"trend + A*Gompertz(k,x0) + B*x*Gompertz":"trend + A*sigmoid(k,x0) + B*x*sigmoid");
    return m;
}

Model fitChirpModel(const vector<ld>& x,const vector<ld>& y,ld w1,ld w2){
    if(x.size()<12||x.size()!=y.size())return{};
    vector<vector<ld>>A(x.size(),vector<ld>(4));
    for(size_t i=0;i<x.size();++i){ld ph=w1*x[i]+w2*x[i]*x[i];A[i]={1,x[i],sinl(ph),cosl(ph)};}
    vector<ld>c;if(!leastSquaresQR(A,y,c))return{};
    Model m;m.valid=true;m.name="Quadratic-phase Fourier / chirp regression";m.category="Spectral / chirp signal model";m.complexity=4;
    m.predict=[c,w1,w2](ld z){ld ph=w1*z+w2*z*z;return c[0]+c[1]*z+c[2]*sinl(ph)+c[3]*cosl(ph);};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y=c0+c1*x+c2*sin(w1*x+w2*x^2)+c3*cos(w1*x+w2*x^2), w1="+fmt(w1,6)+", w2="+fmt(w2,6);return m;
}

// Exponentially decaying (or growing) oscillation: y = offset + exp(-alpha*x) * (A sin(w x) + B cos(w x)).
// This is the standard transient-response shape for a second-order linear system --
// an RLC circuit's ring-down, a damped mechanical oscillator, any underdamped
// step/impulse response -- so it belongs in the family list alongside the existing
// Prony/Z-transform and Fourier signal-processing families rather than only being
// approximated piecewise. alpha and w are supplied by the caller (grid-searched in
// makeSpecs, same pattern as the existing chirp/holt grid searches); once fixed the
// remaining coefficients are a plain linear solve, so this stays fast and exact
// for its basis like every other family here.
Model fitDampedSinusoidModel(const vector<ld>& x,const vector<ld>& y,ld w,ld alpha){
    if(x.size()<10||x.size()!=y.size()||w<=0)return{};
    vector<vector<ld>>A(x.size(),vector<ld>(3));
    for(size_t i=0;i<x.size();++i){
        ld env=expl(clamp(-alpha*x[i],-40.0L,40.0L));
        A[i]={1,env*sinl(w*x[i]),env*cosl(w*x[i])};
    }
    vector<ld>c;if(!leastSquaresQR(A,y,c))return{};
    Model m;m.valid=true;m.name="Damped sinusoid / transient response";
    m.category="Signal processing / electrical transient (RLC-style ring-down)";m.complexity=4;
    m.predict=[c,w,alpha](ld z){ld env=expl(clamp(-alpha*z,-40.0L,40.0L));return c[0]+env*(c[1]*sinl(w*z)+c[2]*cosl(w*z));};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    m.formula="y = "+fmt(c[0])+" + exp(-"+fmt(alpha,6)+"x)*("+fmt(c[1])+" sin("+fmt(w,6)+"x) + "+fmt(c[2])+" cos("+fmt(w,6)+"x))";
    m.note="Second-order underdamped transient-response shape (RLC ring-down / damped oscillator); alpha is the decay rate, w the ringing frequency.";
    return m;
}


Model fitAdaptiveChirpEnvelopeModel(const vector<ld>&x,const vector<ld>&y){
    if(x.size()!=y.size()||x.size()<20)return{};
    auto turns=observedTurningPoints(x,y);
    if(turns.size()<7)return{};
    // Estimate a quadratic phase directly from alternating extrema.  If extrema
    // arrive faster over time, k>0 is a chirp; if they spread out, k<0.
    const ld xm=(x.front()+x.back())*0.5L;
    vector<vector<ld>>PA; vector<ld>pb;
    for(size_t j=0;j<turns.size();++j){ld t=turns[j]-xm;PA.push_back({1.0L,t,0.5L*t*t});pb.push_back((ld)j*PI);}
    vector<ld>pc;if(!leastSquaresQR(PA,pb,pc)||pc.size()!=3)return{};
    ld w0=pc[1], k=pc[2];
    if(!(w0>0)||!isfinite(k))return{};
    const ld span=max<ld>(x.back()-x.front(),1.0L);
    const ld tscale=max<ld>((x.back()-x.front())*0.5L,1.0L);
    const ld knorm=max<ld>(1.0L/(span*span),1e-8L);
    ld best=numeric_limits<ld>::infinity(),bestW=w0,bestK=k; vector<ld>bestC;
    // Small deterministic grid around the calculus-derived phase estimate.
    for(int iw=-4;iw<=4;++iw){
        ld wc=w0*(1.0L+0.05L*iw);
        if(wc<=0)continue;
        for(int ik=-5;ik<=5;++ik){
            ld kc=k + (ld)ik*knorm*0.20L;
            vector<vector<ld>>A(x.size(),vector<ld>(11));
            for(size_t i=0;i<x.size();++i){
                ld t=(x[i]-xm)/tscale;
                ld q=x[i]-xm;
                ld ph=wc*q+0.5L*kc*q*q;
                ld sn=sinl(ph),co=cosl(ph),t2=t*t,t3=t2*t;
                A[i][0]=1;A[i][1]=t;A[i][2]=t2;A[i][3]=t3;
                A[i][4]=sn;A[i][5]=t*sn;A[i][6]=t2*sn;A[i][7]=co;A[i][8]=t*co;A[i][9]=t2*co;A[i][10]=t3*co;
            }
            vector<ld>c;if(!leastSquaresQR(A,y,c))continue;
            ld sse=0;for(size_t i=0;i<x.size();++i){ld p=0;for(int j=0;j<11;++j)p+=A[i][j]*c[j];ld e=y[i]-p;sse+=e*e;}
            ld rms=sqrtl(sse/(ld)x.size());
            // Penalize a phase law that strongly contradicts the observed extrema.
            ld phaseErr=0; int cnt=0;
            for(size_t j=0;j<turns.size();++j){ld q=turns[j]-xm;ld pred=wc*q+0.5L*kc*q*q;ld target=(ld)j*PI;ld e=sinl(0.5L*(pred-target));phaseErr+=fabsl(e);++cnt;}
            ld score=rms + (cnt?phaseErr/(ld)cnt:0)*max<ld>(robustScale(y)*0.10L,1e-6L);
            if(score<best){best=score;bestW=wc;bestK=kc;bestC=c;}
        }
    }
    if(bestC.empty()||!isfinite(best))return{};
    const ld wc=bestW,kc=bestK;
    Model m;m.valid=true;m.exact=false;m.name="Extrema-guided adaptive chirp / evolving-period model";
    m.category="Calculus / nonstationary oscillatory signal model";m.complexity=11;
    m.predict=[c=bestC,wc,kc,xm,tscale](ld z){ld t=(z-xm)/tscale,q=z-xm,ph=wc*q+0.5L*kc*q*q,sn=sinl(ph),co=cosl(ph);return c[0]+c[1]*t+c[2]*t*t+c[3]*t*t*t+c[4]*sn+c[5]*t*sn+c[6]*t*t*sn+c[7]*co+c[8]*t*co+c[9]*t*t*co+c[10]*t*t*t*co;};
    m.fitted.reserve(x.size());for(ld z:x)m.fitted.push_back(m.predict(z));m.nextValue=m.predict(x.back()+1);
    ld periodLast=2*PI/max<ld>(wc+kc*(x.back()-xm),1e-6L);
    ostringstream os;os<<"polynomial baseline + evolving-period Fourier envelope; phase="<<fmt(wc,6)<<"*(x-xmid)+0.5*"<<fmt(kc,6)<<"*(x-xmid)^2; instantaneous period near endpoint≈"<<fmt(periodLast,5);m.formula=os.str();
    m.note="Extrema-guided nonstationary oscillation: turning-point spacing estimates frequency drift while local sine/cosine envelopes absorb amplitude and baseline evolution. This is a heuristic extrapolator unless rolling validation confirms it.";
    return m;
}

Model fitOscillatoryEnvelopeModel(const vector<ld>&x,const vector<ld>&y,ld period){
    if(x.size()!=y.size()||x.size()<16||!(period>2))return{};
    auto turns=observedTurningPoints(x,y);if(turns.size()<5)return{};
    ld baseHalf=max<ld>(period*0.5L,1e-6L), span=max<ld>(x.back()-x.front(),1e-9L), xref=x.back();
    // Find the latest observed extremum and anchor cosine phase there.  This is
    // the key difference from a blind Fourier fit: the phase is tied to actual
    // maxima/minima and the amplitude is allowed to evolve with the domain.
    ld turnX=turns.back();
    auto yAt=[&](ld z){if(z<=x.front())return y.front();if(z>=x.back())return y.back();auto it=lower_bound(x.begin(),x.end(),z);size_t j=(size_t)(it-x.begin());if(j==0)return y[0];ld q=(z-x[j-1])/(x[j]-x[j-1]);return y[j-1]*(1-q)+y[j]*q;};
    ld prevX=turns.size()>=2?turns[turns.size()-2]:turnX- baseHalf, sign=(yAt(turnX)>=yAt(prevX)?1.0L:-1.0L);
    ld spanT=max<ld>(x.back()-x.front(),1.0L), best=numeric_limits<ld>::infinity();vector<ld>bestC;ld bestP=period;
    for(int k=0;k<=24;++k){ld p=period*(0.72L+0.025L*k);if(p<2.0L||p>max<ld>(span*1.2L,3.0L))continue;ld w=2*PI/p;
        vector<vector<ld>>A(x.size(),vector<ld>(7));
        for(size_t i=0;i<x.size();++i){ld t=(x[i]-xref)/spanT,ph=w*(x[i]-turnX),c=sign*cosl(ph);A[i][0]=1;A[i][1]=t;A[i][2]=t*t;A[i][3]=c;A[i][4]=t*c;A[i][5]=t*t*c;A[i][6]=sinl(ph);}
        vector<ld>c;if(!leastSquaresQR(A,y,c))continue;ld sse=0;for(size_t i=0;i<x.size();++i){ld p0=0;for(int j=0;j<7;++j)p0+=A[i][j]*c[j];ld e=y[i]-p0;sse+=e*e;}
        ld rms=sqrtl(sse/(ld)x.size());
        // Penalize periods that disagree strongly with observed turning-point spacing.
        ld periodPenalty=0;if(turns.size()>=5){vector<ld>dd;for(size_t i=1;i<turns.size();++i)dd.push_back(turns[i]-turns[i-1]);ld med=quantile(dd,.5L);periodPenalty=fabsl(p-2*med)/max<ld>(2*med,1e-9L)*max<ld>(robustScale(y)*0.15L,1e-6L);}
        ld score=rms+periodPenalty;if(score<best){best=score;bestC=c;bestP=p;}
    }
    if(bestC.empty()||!isfinite(best))return{};
    const ld w=2*PI/bestP, turnAnchor=turnX, phaseSign=sign;
    Model m;m.valid=true;m.exact=false;m.name="Turning-point guided oscillatory envelope";m.category="Calculus / extrema-aware oscillatory extrapolation";m.complexity=8;
    m.predict=[c=bestC,w,turnAnchor,phaseSign,xref,spanT](ld z){ld t=(z-xref)/spanT,ph=w*(z-turnAnchor),cc=phaseSign*cosl(ph);return c[0]+c[1]*t+c[2]*t*t+(c[3]+c[4]*t+c[5]*t*t)*cc+c[6]*sinl(ph);};
    m.fitted.reserve(x.size());for(ld z:x)m.fitted.push_back(m.predict(z));m.nextValue=m.predict(x.back()+1);
    ostringstream os;os<<"y = quadratic baseline + evolving cosine envelope + phase correction; period≈"<<fmt(bestP,6)<<", phase anchored at turning point x="<<fmt(turnAnchor,5);m.formula=os.str();
    m.note="Shape-aware fallback: uses numerically detected extrema to anchor phase, estimates a cycle length, and fits a changing amplitude/baseline. Forecasts remain heuristic when no globally stable model passes rolling validation.";
    return m;
}


// Detect piecewise-linear / step-response regimes directly from numerical
// derivatives.  This is intentionally separate from the generic piecewise
// polynomial fitter: electrical/control signals often spend many samples in a
// near-constant or constant-slope regime, then switch abruptly to another rule.
// We identify those switches from the first derivative (slope) and fit each
// domain independently.  A final regime is extrapolated using its local rule.
Model fitPiecewiseLinearRegimeModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()!=y.size() || y.size()<10) return {};
    const int n=sizeToInt(y.size());
    vector<ld>d; d.reserve(n-1);
    for(int i=1;i<n;++i){
        ld dx=x[i]-x[i-1];
        if(fabsl(dx)<1e-15L) return {};
        d.push_back((y[i]-y[i-1])/dx);
    }
    if(d.size()<6) return {};

    // Derivative-change signal.  Piecewise-linear rules generate impulses here,
    // while a smooth nonlinear curve normally changes slope gradually.
    vector<ld>dd; dd.reserve(d.size()-1);
    for(size_t i=1;i<d.size();++i) dd.push_back(d[i]-d[i-1]);
    ld ddMed=quantile(dd,.5L), ddNoise=0;
    vector<ld>adev; adev.reserve(dd.size());
    for(ld q:dd) adev.push_back(fabsl(q-ddMed));
    ddNoise=max<ld>(1.4826L*quantile(adev,.5L),1e-12L);
    ld yScale=max<ld>(robustScale(y),1e-12L);

    struct BC { int idx; ld score; };
    vector<BC> breaks;
    // A true piecewise-linear signal changes its first derivative abruptly.
    // Inspect adjacent slope regimes directly, then confirm the change against
    // short one-sided medians. This preserves single-sample hold domains such as
    // ramp -> hold -> fall -> hold -> ramp, which a wide moving window can smear.
    for(int i=1;i<(int)d.size();++i){
        ld immediate=fabsl(d[i]-d[i-1]);
        int lw=min(3,i), rw=min(3,(int)d.size()-i);
        vector<ld>L(d.begin()+i-lw,d.begin()+i), R(d.begin()+i,d.begin()+i+rw);
        if(L.empty()||R.empty()) continue;
        ld lm=quantile(L,.5L), rm=quantile(R,.5L), delta=fabsl(rm-lm);
        // Estimate noise inside each side separately. Mixing both sides into
        // one scale can hide a real ramp->hold or hold->ramp boundary because
        // the two different levels inflate the apparent noise.
        ld leftNoise=max<ld>(robustScale(L),1e-12L);
        ld rightNoise=max<ld>(robustScale(R),1e-12L);
        ld localScale=max(leftNoise,rightNoise);
        ld threshold=max({5.0L*ddNoise,4.0L*localScale,0.025L*max({fabsl(lm),fabsl(rm),1.0L}),0.001L*yScale});
        const ld eventThreshold=max({8.0L*ddNoise,0.02L*max(yScale,1.0L),0.5L*localScale});
        if(immediate>eventThreshold || delta>threshold){
            breaks.push_back({i, max(immediate,delta)/max<ld>(min(eventThreshold,threshold),1e-12L)});
        }
    }
    sort(breaks.begin(),breaks.end(),[](const BC&a,const BC&b){return a.idx<b.idx;});
    vector<BC> clustered;
    // Adjacent slope changes can describe a short hold domain. Do not merge
    // them unless they are literally adjacent measurements of the same event.
    for(const auto&b:breaks){
        if(clustered.empty() || b.idx-clustered.back().idx>0) clustered.push_back(b);
        else if(b.score>clustered.back().score) clustered.back()=b;
    }
    if(clustered.empty()) return {};
    // Keep the strongest few boundaries; this guards runtime without limiting the
    // model to only one or two domains.
    sort(clustered.begin(),clustered.end(),[](const BC&a,const BC&b){return a.score>b.score;});
    if(clustered.size()>6) clustered.resize(6);
    sort(clustered.begin(),clustered.end(),[](const BC&a,const BC&b){return a.idx<b.idx;});

    // Candidate cuts include only derivative-change events, not arbitrary deciles.
    // Search up to four actual boundaries; eight or more domains from a 30-point
    // signal is more often overfitting than a useful piecewise rule.
    vector<int> cands; for(const auto&b:clustered) cands.push_back(b.idx);
    vector<vector<int>> sets;
    for(int c:cands) sets.push_back({c});
    for(size_t i=0;i<cands.size();++i) for(size_t j=i+1;j<cands.size();++j) sets.push_back({cands[i],cands[j]});
    for(size_t i=0;i<cands.size();++i) for(size_t j=i+1;j<cands.size();++j) for(size_t k=j+1;k<cands.size();++k) sets.push_back({cands[i],cands[j],cands[k]});
    if(cands.size()<=6) for(size_t i=0;i<cands.size();++i) for(size_t j=i+1;j<cands.size();++j) for(size_t k=j+1;k<cands.size();++k) for(size_t q=k+1;q<cands.size();++q) sets.push_back({cands[i],cands[j],cands[k],cands[q]});

    auto fitSegment=[&](int lo,int hi,Model&out)->bool{
        if(hi-lo<1) return false;
        if(hi-lo==1){
            const ld dx=x[hi]-x[lo];
            if(fabsl(dx)<1e-18L) return false;
            const ld slope=(y[hi]-y[lo])/dx;
            const ld intercept=y[lo]-slope*x[lo];
            out.valid=true;out.name="Local linear regime";out.category="Piecewise linear regime";out.complexity=2;
            out.formula="y = "+fmt(intercept)+" + ("+fmt(slope)+")x";
            out.predict=[intercept,slope](ld z){return intercept+slope*z;};
            out.fitted={y[lo],y[hi]};out.nextValue=out.predict(x.back()+1.0L);
            return true;
        }
        vector<ld>tx(x.begin()+lo,x.begin()+hi+1), ty(y.begin()+lo,y.begin()+hi+1);
        out=fitLinearModel(tx,ty);
        return out.valid&&out.predict;
    };
    ld bestScore=numeric_limits<ld>::infinity(); vector<int>bestBreaks; vector<Model>bestModels;
    // Baseline single-line BIC for a meaningful comparison.
    Model whole=fitLinearModel(x,y); if(!whole.valid||!whole.predict) return {};
    auto [wholeRMSE,wholeMAE]=errorMetrics(y,whole.fitted);
    (void)wholeMAE;
    const ld wholeSSE=wholeRMSE*wholeRMSE*(ld)n;
    const ld nlog=max<ld>(logl((ld)n),1.0L);
    for(auto br:sets){
        sort(br.begin(),br.end()); bool ok=true;
        for(size_t j=1;j<br.size();++j) if(br[j]-br[j-1]<1) ok=false;
        if(!ok) continue;
        vector<int>cuts={0}; cuts.insert(cuts.end(),br.begin(),br.end()); cuts.push_back(n-1);
        vector<Model>mods; ld sse=0;
        for(size_t j=0;j+1<cuts.size();++j){
            Model m; if(!fitSegment(cuts[j],cuts[j+1],m)){ok=false;break;}
            int lo=cuts[j], hi=cuts[j+1];
            for(int i=lo;i<=hi;++i){ld e=y[i]-m.predict(x[i]);sse+=e*e;}
            mods.push_back(m);
        }
        if(!ok||mods.size()<2) continue;
        // 2 parameters per linear segment + a mild complexity penalty.
        int params=2*sizeToInt(mods.size());
        ld bic=(ld)n*logl(max<ld>(sse/(ld)n,1e-18L))+params*nlog;
        // Reject cosmetic segmentations that do not materially improve the fit.
        if(sse>0.70L*wholeSSE && sse>1e-16L*max<ld>(yScale*yScale*n,1.0L)) continue;
        if(bic<bestScore){bestScore=bic;bestBreaks=br;bestModels=mods;}
    }
    if(bestModels.size()<2) return {};

    struct Part{ld lo=0,hi=0;Model model;};
    vector<Part>parts; parts.reserve(bestModels.size());
    for(size_t j=0;j<bestModels.size();++j){
        int lo=(j==0?0:bestBreaks[j-1]);
        int hi=(j<bestBreaks.size()?bestBreaks[j]:n-1);
        // At a domain boundary the right-hand rule owns the boundary point.
        parts.push_back({x[lo],x[hi],bestModels[j]});
    }
    Model m; m.valid=true; m.exact=false; m.name="Derivative-segmented piecewise-linear signal model";
    m.category="Calculus / piecewise-linear signal-regime model"; m.complexity=2*sizeToInt(parts.size());
    m.predict=[parts](ld z){
        size_t k=0;
        while(k+1<parts.size() && z>=parts[k+1].lo) ++k;
        return parts[k].model.predict(z);
    };
    m.fitted.reserve(n); for(ld z:x) m.fitted.push_back(m.predict(z));
    m.nextValue=m.predict(x.back()+1.0L);
    ostringstream os; os<<"piecewise linear/hold domains selected from derivative regime changes: ";
    for(size_t j=0;j<parts.size();++j){
        if(j) os<<" | ";
        os<<"["<<fmt(parts[j].lo,5)<<","<<fmt(parts[j].hi,5)<<"] -> "<<parts[j].model.formula;
    }
    m.formula=os.str();
    m.note="Numerical first derivatives were clustered into locally stable regimes. Breakpoints represent changes in slope, including flat/hold domains and discontinuous jumps; the final detected regime is extrapolated forward.";
    return m;
}

Model fitPiecewiseConstantModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<8||x.size()!=y.size())return{};
    vector<int> starts={0};
    ld diffScale=0;vector<ld>d;d.reserve(y.size()-1);for(size_t i=1;i<y.size();++i)d.push_back(y[i]-y[i-1]);
    diffScale=max<ld>(1.4826L*quantile([&](){vector<ld>ad;ld md=quantile(d,.5L);for(ld q:d)ad.push_back(fabsl(q-md));return ad;}(),.5L),1e-9L);
    int start=0;
    while(start<(int)y.size()-1){
        int next=-1;
        for(int i=start+1;i<(int)y.size();++i){
            ld localMed=quantile(vector<ld>(y.begin()+start,y.begin()+i),.5L);
            if(fabsl(y[i]-localMed)>2.5L*diffScale){
                int runEnd=min((int)y.size(),i+2);
                ld newMed=quantile(vector<ld>(y.begin()+i,y.begin()+runEnd),.5L);
                if(fabsl(newMed-localMed)>2.0L*diffScale){next=i;break;}
            }
        }
        if(next<0)break;
        if(next-start<2){start++;continue;}
        starts.push_back(next);start=next;
        if(starts.size()>8)break;
    }
    if(starts.size()<2)return{};
    starts.push_back(sizeToInt(y.size()));
    vector<ld>levels;
    for(size_t s=0;s+1<starts.size();++s){int a=starts[s],b=starts[s+1];vector<ld>seg(y.begin()+a,y.begin()+b);levels.push_back(quantile(seg,.5L));}
    ld last=levels.back();
    Model m;m.valid=true;m.name="Robust piecewise-constant / zero-order-hold model";m.category="Signal processing / segmentation";m.complexity=sizeToInt(levels.size());
    m.predict=[last](ld){return last;};m.nextValue=last;m.fitted.reserve(y.size());for(size_t i=0;i<y.size();++i){size_t k=upper_bound(starts.begin(),starts.end(),(int)i)-starts.begin();k=max<size_t>(1,k)-1;m.fitted.push_back(levels[min(k,levels.size()-1)]);}
    ostringstream os;os<<"piecewise constant levels: ";for(size_t i=0;i<levels.size();++i){if(i)os<<" | ";os<<fmt(levels[i],5);}m.formula=os.str();m.note="Robust plateau segmentation with local median levels; the final plateau is held forward.";return m;
}

Model fitSegmentedLinearModel(const vector<ld>& x,const vector<ld>& y,int breaks){
    if(x.size()<12||x.size()!=y.size()||breaks<1||breaks>3)return{};
    vector<ld> candidates;
    auto cps=detectChangePoints(y,4,2.5L);
    for(int c:cps) if(c>=4&&c+4<(int)x.size()) candidates.push_back(x[c]);
    for(int q=1;q<=9;++q){int idx=(int)llround((ld)q*x.size()/10.0L);if(idx>=4&&idx+4<(int)x.size())candidates.push_back(x[idx]);}
    sort(candidates.begin(),candidates.end());candidates.erase(unique(candidates.begin(),candidates.end(),[](ld a,ld b){return fabsl(a-b)<1e-10L;}),candidates.end());
    if((int)candidates.size()<breaks)return{};
    vector<ld> bestB;ld best=numeric_limits<ld>::infinity();vector<ld>bestC;
    auto fitFor=[&](const vector<ld>&bs){
        vector<vector<ld>>A(x.size(),vector<ld>(2+bs.size()));
        for(size_t i=0;i<x.size();++i){A[i][0]=1;A[i][1]=x[i];for(size_t j=0;j<bs.size();++j)A[i][2+j]=max<ld>(0,x[i]-bs[j]);}
        vector<ld>c;if(!leastSquaresQR(A,y,c))return pair<ld,vector<ld>>{numeric_limits<ld>::infinity(),{}};
        ld sse=0;for(size_t i=0;i<x.size();++i){ld p=c[0]+c[1]*x[i];for(size_t j=0;j<bs.size();++j)p+=c[2+j]*max<ld>(0,x[i]-bs[j]);ld e=y[i]-p;sse+=e*e;}return pair<ld,vector<ld>>{sse,c};
    };
    for(ld b:candidates){auto fc=fitFor({b});if(breaks==1&&fc.first<best){best=fc.first;bestB={b};bestC=fc.second;}}
    if(breaks>=2){for(size_t i=0;i<candidates.size();++i)for(size_t j=i+1;j<candidates.size();++j){auto fc=fitFor({candidates[i],candidates[j]});if(fc.first<best){best=fc.first;bestB={candidates[i],candidates[j]};bestC=fc.second;}}}
    if(breaks>=3&&candidates.size()<=30){for(size_t i=0;i<candidates.size();++i)for(size_t j=i+1;j<candidates.size();++j)for(size_t k=j+1;k<candidates.size();++k){auto fc=fitFor({candidates[i],candidates[j],candidates[k]});if(fc.first<best){best=fc.first;bestB={candidates[i],candidates[j],candidates[k]};bestC=fc.second;}}}
    if(bestC.empty())return{};
    sort(bestB.begin(),bestB.end());
    Model m;m.valid=true;m.name="Segmented linear regression";m.category="Signal processing / piecewise regression";m.complexity=2+sizeToInt(bestB.size());
    m.predict=[bestC,bestB](ld z){ld p=bestC[0]+bestC[1]*z;for(size_t j=0;j<bestB.size();++j)p+=bestC[2+j]*max<ld>(0,z-bestB[j]);return p;};m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));
    ostringstream os;os<<"continuous piecewise-linear hinge model; breaks=";for(size_t i=0;i<bestB.size();++i){if(i)os<<",";os<<fmt(bestB[i],6);}m.formula=os.str();return m;
}

Model fitPiecewiseLinearModel(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<10)return{};
    ld xmin=*min_element(x.begin(),x.end()),xmax=*max_element(x.begin(),x.end()),span=max<ld>(xmax-xmin,1e-9L);
    ld best=numeric_limits<ld>::infinity();vector<ld>bc;ld bb=xmin+span*.5;
    for(int q=2;q<=8;++q){
        ld br=xmin+span*(q/10.0L);
        vector<vector<ld>>A(x.size(),vector<ld>(3));
        for(size_t i=0;i<x.size();++i){A[i][0]=1;A[i][1]=x[i];A[i][2]=max<ld>(0,x[i]-br);}
        vector<ld>c;if(!leastSquaresQR(A,y,c))continue;ld sse=0;for(size_t i=0;i<x.size();++i){ld p=c[0]+c[1]*A[i][1]+c[2]*A[i][2];ld e=y[i]-p;sse+=e*e;}if(sse<best){best=sse;bc=c;bb=br;}
    }
    if(bc.empty())return{};
    Model m;m.valid=true;m.name="Piecewise linear / change-point regression";m.category="Change-point regression";m.complexity=4;
    m.predict=[c=bc,br=bb](ld z){return c[0]+c[1]*z+c[2]*max<ld>(0,z-br);};
    m.nextValue=m.predict(x.back()+1);m.fitted.reserve(x.size());for(ld xx:x)m.fitted.push_back(m.predict(xx));m.formula="y=a+b*x+c*max(0,x-break), break="+fmt(bb,5);return m;
}

struct SeasonalTemplateState {
    ld x0=0;
    int period=0;
    int trendDegree=0;
    vector<ld> coef; // intercept + trend terms + p-1 phase offsets
};

Model fitSeasonalTemplateModel(const vector<ld>&x,const vector<ld>&y,int period,int trendDegree){
    if(x.size()!=y.size() || x.size()<10 || !isUnitSpacedGrid(x) || period<2 || period>(int)y.size()/2 || trendDegree<0 || trendDegree>2) return {};
    const int n=sizeToInt(y.size()), p=period;
    const int cols=1+trendDegree+(p-1);
    vector<vector<ld>>A(n,vector<ld>(cols,0));
    ld x0=x.front();
    ld span=max<ld>(x.back()-x.front(),1.0L);
    for(int i=0;i<n;++i){
        ld t=(x[i]-x0)/span;
        A[i][0]=1;
        ld q=t;
        for(int d=1;d<=trendDegree;++d){A[i][d]=q;q*=t;}
        long long absIndex=llroundl(x[i]);
        int phase=(int)((absIndex%p+p)%p);
        if(phase>0) A[i][1+trendDegree+(phase-1)]=1;
    }
    vector<ld>c;if(!leastSquaresQR(A,y,c))return {};
    Model m;m.valid=true;m.name="Seasonal template / periodic regression (period "+to_string(p)+")";
    m.category="Periodic / seasonal sequence model";m.complexity=cols+1;
    m.predict=[c,x0,span,p,trendDegree](ld z){
        ld t=(z-x0)/span,base=c[0];ld q=t;
        for(int d=1;d<=trendDegree;++d){base+=c[d]*q;q*=t;}
        long long idx=llroundl(z);int phase=(int)((idx%p+p)%p);
        if(phase>0 && 1+trendDegree+(phase-1)<(int)c.size()) base+=c[1+trendDegree+(phase-1)];
        return base;
    };
    m.fitted.reserve(n);for(ld z:x)m.fitted.push_back(m.predict(z));
    m.nextValue=m.predict(x.back()+1);
    ostringstream os;os<<"seasonal template period="<<p<<", trend degree="<<trendDegree<<"; phase offsets repeat every "<<p<<" samples";m.formula=os.str();
    m.note="Explicitly separates slow trend from a repeating phase template; useful for step-like periodic, interleaved, and seasonal patterns that a straight line can hide.";
    return m;
}

Model fitPeriodicDifferenceModel(const vector<ld>&x,const vector<ld>&y,int period){
    if(x.size()!=y.size() || x.size()<10 || !isUnitSpacedGrid(x) || period<2 || period>(int)y.size()/2) return {};
    const int n=sizeToInt(y.size()),p=period;
    vector<ld>delta(p,0);vector<int>count(p,0);
    for(int i=p;i<n;++i){int phase=(int)((llroundl(x[i])%p+p)%p);delta[phase]+=y[i]-y[i-p];count[phase]++;}
    for(int ph=0;ph<p;++ph){if(count[ph]==0)return {};delta[ph]/=count[ph];}
    ld err=0,scale=max<ld>(robustScale(y),1e-12L);int cnt=0;
    for(int i=p;i<n;++i){int phase=(int)((llroundl(x[i])%p+p)%p);ld e=(y[i]-y[i-p])-delta[phase];err+=fabsl(e);cnt++;}
    if(cnt==0)return {};
    ld mae=err/cnt;
    // Require the repeating increment structure to be meaningfully smaller
    // than ordinary signal variation; this prevents arbitrary lag recursion.
    if(mae>0.75L*scale)return {};
    const ld x0=x.front();
    Model m;m.valid=true;m.name="Periodic-difference recurrence (period "+to_string(p)+")";m.category="Periodic recurrence / seasonal increments";m.complexity=p+1;
    m.predict=[y,x0,p,delta](ld z){
        long long idx=llroundl(z-x0);if(idx<0)return numeric_limits<ld>::quiet_NaN();
        if(idx<(long long)y.size())return y[(size_t)idx];
        vector<ld>v=y;v.reserve((size_t)idx+1);
        for(long long i=(long long)y.size();i<=idx;++i){long long absIndex=llroundl(x0)+i;int ph=(int)((absIndex%p+p)%p);v.push_back(v[(size_t)(i-p)]+delta[ph]);}
        return v.back();
    };
    m.fitted=y;m.nextValue=m.predict(x.back()+1);
    m.formula="y[n]=y[n-"+to_string(p)+"] + repeating phase increment; period="+to_string(p);
    m.note="Forecast uses the learned increment for each phase of the detected cycle, rather than assuming the whole signal is linear.";
    return m;
}

struct PieceSegment { ld loX=0, hiX=0, mid=0, half=1; int degree=1; vector<ld> coef; };

bool fitPieceSegment(const vector<ld>&x,const vector<ld>&y,int lo,int hi,int degree,PieceSegment&out){
    if(hi-lo<degree+2)return false;
    ld mid=(x[lo]+x[hi-1])/2.0L,half=max<ld>((x[hi-1]-x[lo])/2.0L,1e-9L);
    int cols=degree+1;vector<vector<ld>>A(hi-lo,vector<ld>(cols));vector<ld>b(y.begin()+lo,y.begin()+hi);
    for(int i=lo;i<hi;++i){ld t=(x[i]-mid)/half,q=1;for(int j=0;j<cols;++j){A[i-lo][j]=q;q*=t;}}
    vector<ld>c;if(!leastSquaresQR(A,b,c))return false;
    out.loX=x[lo];out.hiX=x[hi-1];out.mid=mid;out.half=half;out.degree=degree;out.coef=c;return true;
}

ld evalPieceSegment(const PieceSegment&s,ld z){ld t=(z-s.mid)/s.half,q=1,v=0;for(ld c:s.coef){v+=c*q;q*=t;}return v;}

// Local shifted-exponential fit for a piecewise domain.  The ordinary global
// exponential model cannot represent y = c + a*exp(bx), so this local version
// estimates the asymptotic offset c from a small robust grid and then solves
// the remaining log-linear problem exactly.  It is deliberately local: a
// complicated global signal can therefore contain an exponential tail without
// forcing the whole dataset into an exponential family.
Model fitShiftedExponentialLocal(const vector<ld>& x,const vector<ld>& y){
    if(x.size()<7||x.size()!=y.size())return{};
    ld minY=*min_element(y.begin(),y.end()),maxY=*max_element(y.begin(),y.end());
    ld span=max<ld>(maxY-minY,robustScale(y));
    if(span<1e-12L)return{};
    vector<ld> offsets;
    // Fine enough to resolve an unknown vertical asymptote while staying cheap
    // because this search is performed only inside already-detected local pieces.
    for(int k=0;k<=160;++k) offsets.push_back(minY-2.0L*span*(ld)k/160.0L);
    offsets.push_back(minY-3.0L*span);
    ld best=numeric_limits<ld>::infinity(),bestC=0,bestA=0,bestB=0;
    const ld x0=x.front();
    for(ld c:offsets){
        vector<ld>ly(y.size());bool ok=true;
        for(size_t i=0;i<y.size();++i){ld q=y[i]-c;if(!(q>0)&&isfinite(q)){ok=false;break;}ly[i]=logl(max<ld>(q,1e-300L));}
        if(!ok)continue;
        vector<ld>u(x.size());for(size_t i=0;i<x.size();++i)u[i]=x[i]-x0;
        auto coef=linearFit(u,ly);if(coef.size()!=2)continue;
        ld a=expl(coef[0]),b=coef[1];if(!isfinite(a)||!isfinite(b))continue;
        ld sse=0;bool finite=true;
        for(size_t i=0;i<x.size();++i){ld e=b*u[i];ld p;if(e>10000)p=numeric_limits<ld>::infinity();else p=c+a*expl(e);if(!isfinite(p)){finite=false;break;}ld r=y[i]-p;sse+=r*r;}
        if(finite&&sse<best){best=sse;bestC=c;bestA=a;bestB=b;}
    }
    if(!isfinite(best))return{};
    Model m;m.valid=true;m.name="Shifted exponential (local piece)";m.category="Piecewise nonlinear regression";m.complexity=3;
    m.predict=[bestC,bestA,bestB,x0](ld z){ld e=bestB*(z-x0);if(e>10000)return numeric_limits<ld>::infinity();return bestC+bestA*expl(e);};
    m.fitted.reserve(x.size());for(ld z:x)m.fitted.push_back(m.predict(z));
    m.nextValue=m.predict(x.back()+1);
    m.formula="y = c + a exp(b(x-x0)), local c="+fmt(bestC)+", a="+fmt(bestA)+", b="+fmt(bestB)+", x0="+fmt(x0);
    return m;
}

Model fitBestLocalPieceModel(const vector<ld>&x,const vector<ld>&y){
    if(x.size()<7||x.size()!=y.size())return{};
    struct LocalSpec{int kind=0;int degree=1;ld w=0;};
    vector<LocalSpec>specs={{0,1,0},{1,2,0}};
    if(x.size()>=9)specs.push_back({2,3,0});
    specs.push_back({3,0,0});
    if(x.size()>=8){
        ld span=max<ld>(x.back()-x.front(),1e-9L);
        for(int cycles=1;cycles<=3;++cycles){
            ld period=max<ld>(span/(ld)cycles*2.0L,2.5L);
            specs.push_back({4,0,2*PI/period});
        }
    }
    auto fitSpec=[&](const LocalSpec&sp,const vector<ld>&tx,const vector<ld>&ty)->Model{
        if(sp.kind==0)return fitLinearModel(tx,ty);
        if(sp.kind==1)return fitPolyModel(tx,ty,2);
        if(sp.kind==2)return fitPolyModel(tx,ty,3);
        if(sp.kind==3)return fitShiftedExponentialLocal(tx,ty);
        if(sp.kind==4)return fitSinusoidModel(tx,ty,sp.w);
        return {};
    };
    ld scale=max<ld>(robustScale(y),1e-12L);ld bestScore=numeric_limits<ld>::infinity();Model best;
    const int n=sizeToInt(y.size());
    const int trainN=(n>=10)?max(6,(int)floorl(0.70L*n)):n;
    for(const auto&sp:specs){
        Model full=fitSpec(sp,x,y);if(!full.valid||!full.predict)continue;
        ld trainSSE=0;bool trainFinite=true;
        for(size_t i=0;i<x.size();++i){ld p=full.predict(x[i]);if(!isfinite(p)){trainFinite=false;break;}ld e=y[i]-p;trainSSE+=e*e;}
        if(!trainFinite)continue;
        ld score;
        if(trainN<n){
            vector<ld>tx(x.begin(),x.begin()+trainN),ty(y.begin(),y.begin()+trainN);
            Model hold=fitSpec(sp,tx,ty);if(!hold.valid||!hold.predict)continue;
            ld valSSE=0;bool valFinite=true;int valN=n-trainN;
            for(int i=trainN;i<n;++i){ld p=hold.predict(x[i]);if(!isfinite(p)){valFinite=false;break;}ld e=y[i]-p;valSSE+=e*e;}
            if(!valFinite)continue;
            ld valMSE=valSSE/(ld)max(1,valN);ld trainMSE=trainSSE/(ld)n;
            score=valMSE/(scale*scale)+0.05L*max(1,sp.kind==0?2:full.complexity)*logl((ld)n)/(ld)n+0.02L*trainMSE/(scale*scale);
        }else{
            score=(ld)n*logl(max<ld>(trainSSE/(ld)n,1e-18L))+max(1,full.complexity)*logl((ld)n);
        }
        if(score<bestScore){bestScore=score;best=full;}
    }
    return best;
}

vector<int> structuralBreakCandidates(const vector<ld>&x,const vector<ld>&y){
    vector<int>cand;int n=sizeToInt(y.size());if(n<12)return cand;
    auto cps=detectChangePoints(y,4,2.2L);for(int c:cps)if(c>=4&&c<=n-5)cand.push_back(c);
    vector<ld>slope;for(int i=1;i<n;++i){ld dx=x[i]-x[i-1];slope.push_back(fabsl(dx)>1e-18L?(y[i]-y[i-1])/dx:0);}
    ld s=max<ld>(robustScale(slope),1e-9L);
    const ld yScale=max<ld>(robustScale(y),1e-9L);
    for(int i=4;i<n-4;++i){
        // Finite-difference approximation to the left/right derivative limits.
        // A persistent mismatch means the function's local rule changed at i;
        // a one-step jump is treated as a possible discontinuity.
        ld left=meanOf(vector<ld>(slope.begin()+max(0,i-4),slope.begin()+i));
        ld right=meanOf(vector<ld>(slope.begin()+i,slope.begin()+min((int)slope.size(),i+4)));
        ld localNoise=max<ld>(s,medianAbs(slope));
        if(fabsl(right-left)>2.0L*max<ld>(localNoise,1e-9L))cand.push_back(i);
        ld jump=fabsl(y[i]-y[i-1]);
        if(jump>4.0L*yScale)cand.push_back(i);
        if(i>=2&&i+2<n){
            ld leftNear=(slope[i-1]+slope[i-2])*0.5L;
            ld rightNear=(slope[i]+slope[i+1])*0.5L;
            if(fabsl(rightNear-leftNear)>2.5L*max<ld>(localNoise,1e-9L))cand.push_back(i);
        }
    }
    sort(cand.begin(),cand.end());cand.erase(unique(cand.begin(),cand.end()),cand.end());
    if(cand.size()>18){
        vector<int>reduced;reduced.reserve(18);
        for(int q=0;q<18;++q){size_t idx=(size_t)llround((ld)q*(cand.size()-1)/17.0L);reduced.push_back(cand[idx]);}
        sort(reduced.begin(),reduced.end());reduced.erase(unique(reduced.begin(),reduced.end()),reduced.end());cand.swap(reduced);
    }
    return cand;
}

Model fitPiecewiseAdaptiveModel(const vector<ld>&x,const vector<ld>&y,int requestedDegree){
    if(x.size()!=y.size() || x.size()<14 || requestedDegree<1 || requestedDegree>2)return {};
    const int n=sizeToInt(y.size());auto cand=structuralBreakCandidates(x,y);
    // A gradual change (for example linear -> exponential) may not generate a
    // sharp derivative spike. Add a small coarse set of possible domain cuts so
    // the local-family selector still gets a chance to discover it.
    const int mid=n/2;
    if(mid>=5&&mid<=n-5)cand.push_back(mid);
    if(cand.empty()){
        const int stride=max(2,(n-10)/8);
        for(int c=5;c<=n-5;c+=stride)cand.push_back(c);
    }
    sort(cand.begin(),cand.end());cand.erase(unique(cand.begin(),cand.end()),cand.end());
    if(cand.empty())return {};
    vector<int> bestBreaks;vector<PieceSegment>bestSegs;ld bestScore=numeric_limits<ld>::infinity();int bestDegree=requestedDegree;
    vector<vector<int>> sets;for(int c:cand)sets.push_back({c});
    for(size_t i=0;i<cand.size();++i)for(size_t j=i+1;j<cand.size();++j)sets.push_back({cand[i],cand[j]});
    if(cand.size()<=24)for(size_t i=0;i<cand.size();++i)for(size_t j=i+1;j<cand.size();++j)for(size_t k=j+1;k<cand.size();++k)sets.push_back({cand[i],cand[j],cand[k]});
    ld scale=max<ld>(varianceOf(y,false),1e-12L);
    for(auto br:sets){
        sort(br.begin(),br.end());bool ok=true;for(size_t i=1;i<br.size();++i)if(br[i]-br[i-1]<5)ok=false;if(!ok)continue;
        vector<int>cuts={0};cuts.insert(cuts.end(),br.begin(),br.end());cuts.push_back(n);
        vector<PieceSegment>segs;ld sse=0;for(size_t q=0;q+1<cuts.size();++q){PieceSegment sg;if(!fitPieceSegment(x,y,cuts[q],cuts[q+1],requestedDegree,sg)){ok=false;break;}for(int i=cuts[q];i<cuts[q+1];++i){ld e=y[i]-evalPieceSegment(sg,x[i]);sse+=e*e;}segs.push_back(sg);}if(!ok)continue;
        int params=sizeToInt(segs.size())*(requestedDegree+1);ld bic=(ld)n*logl(max<ld>(sse/(ld)n,1e-18L))+params*logl((ld)n)+0.5L*params*scale;
        if(bic<bestScore){bestScore=bic;bestBreaks=br;bestSegs=segs;bestDegree=requestedDegree;}
    }
    if(bestSegs.size()<2)return {};

    struct HybridPart{ld loX=0,hiX=0;Model model;};
    vector<HybridPart>parts;parts.reserve(bestSegs.size());
    vector<int>finalCuts={0};finalCuts.insert(finalCuts.end(),bestBreaks.begin(),bestBreaks.end());finalCuts.push_back(n);
    for(size_t q=0;q+1<finalCuts.size();++q){
        int lo=finalCuts[q],hi=finalCuts[q+1];
        vector<ld>tx(x.begin()+lo,x.begin()+hi),ty(y.begin()+lo,y.begin()+hi);
        Model lm=fitBestLocalPieceModel(tx,ty);
        if(!lm.valid){
            PieceSegment sg=bestSegs[q];
            lm.valid=true;lm.name="Polynomial local piece";lm.category="Piecewise polynomial";lm.complexity=bestDegree+1;
            lm.predict=[sg](ld z){return evalPieceSegment(sg,z);};
            lm.formula="local polynomial degree "+to_string(bestDegree);
            lm.fitted.reserve(tx.size());for(ld z:tx)lm.fitted.push_back(lm.predict(z));
        }
        // Preserve a polynomial segment when it fits at least as well as a more
        // elaborate local family; this prevents gratuitous nonlinear overfitting.
        parts.push_back({x[lo],x[hi-1],lm});
    }
    if(parts.size()<2)return {};
    Model m;m.valid=true;m.name="Adaptive piecewise function / change-point model";m.category="Piecewise / segmented function model";m.complexity=0;
    for(const auto&p:parts)m.complexity+=max(1,p.model.complexity);
    m.predict=[parts](ld z){
        size_t k=0;
        while(k+1<parts.size()&&z>=parts[k+1].loX)++k;
        // If two domains meet at the same x, the right-hand domain owns that point.
        return parts[k].model.predict(z);
    };
    m.fitted.reserve(n);for(ld z:x)m.fitted.push_back(m.predict(z));m.nextValue=m.predict(x.back()+1);
    ostringstream os;os<<"piecewise function with "<<parts.size()<<" domains: ";
    for(size_t i=0;i<parts.size();++i){if(i)os<<" | ";os<<"["<<fmt(parts[i].loX,4)<<","<<fmt(parts[i].hiX,4)<<"] -> "<<parts[i].model.name;}
    m.formula=os.str();
    m.note="Derivative/change-point diagnostics partition the domain; each segment chooses its own validated local functional family, allowing slope changes and discontinuities, with the final segment extrapolated forward.";
    return m;
}

vector<int> strongShapeBreakCandidates(const vector<ld>&x,const vector<ld>&y){
    vector<pair<int,ld>> scored;
    if(x.size()!=y.size()||x.size()<12)return {};
    vector<ld>d;d.reserve(y.size()-1);
    for(size_t i=1;i<y.size();++i){ld dx=x[i]-x[i-1];d.push_back(fabsl(dx)>1e-15L?(y[i]-y[i-1])/dx:0.0L);}
    ld globalMed=quantile(d,.5L);vector<ld>dev;for(ld q:d)dev.push_back(fabsl(q-globalMed));
    ld globalNoise=max<ld>(1.4826L*quantile(dev,.5L),1e-9L);
    for(int i=5;i<(int)y.size()-5;++i){
        vector<ld>leftD(d.begin()+i-4,d.begin()+i),rightD(d.begin()+i,d.begin()+i+4);
        ld left=quantile(leftD,.5L),right=quantile(rightD,.5L);
        vector<ld>localD=leftD;localD.insert(localD.end(),rightD.begin(),rightD.end());
        ld localScale=max<ld>(1.4826L*quantile([&](){vector<ld>e;ld mm=quantile(localD,.5L);for(ld q:localD)e.push_back(fabsl(q-mm));return e;}(),.5L),1e-9L);
        ld slopeJump=fabsl(right-left);
        ld jump=fabsl(y[i]-y[i-1]);
        vector<ld>neigh;for(int j=max(0,i-3);j<=i-1;++j)neigh.push_back(fabsl(y[j+1]-y[j]));for(int j=i+1;j<=min((int)y.size()-2,i+3);++j)neigh.push_back(fabsl(y[j+1]-y[j]));
        ld neighborMedian=neigh.empty()?globalNoise:quantile(neigh,.5L);
        bool isolatedJump=jump>max<ld>(6.0L*max<ld>(neighborMedian,1e-9L),4.0L*max<ld>(globalNoise,1e-9L));
        bool derivativeBreak=slopeJump>max<ld>(3.0L*localScale,0.20L*max({fabsl(left),fabsl(right),1.0L}));
        if(isolatedJump||derivativeBreak){
            ld jumpStrength=isolatedJump?(jump/max<ld>(neighborMedian,1e-9L)+10.0L):0.0L;
            ld derivStrength=derivativeBreak?(slopeJump/max<ld>(localScale,1e-9L)):0.0L;
            ld strength=max(jumpStrength,derivStrength);
            int candidate=isolatedJump?i:(i+1);
            scored.push_back({candidate,strength});
        }
    }
    sort(scored.begin(),scored.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    vector<pair<int,ld>> clusters;
    for(auto q:scored){
        if(clusters.empty()||q.first-clusters.back().first>=3)clusters.push_back(q);
        else if(q.second>clusters.back().second)clusters.back()=q;
    }
    sort(clusters.begin(),clusters.end(),[](const auto&a,const auto&b){return a.second>b.second;});
    if(clusters.size()>4)clusters.resize(4);
    sort(clusters.begin(),clusters.end(),[](const auto&a,const auto&b){return a.first<b.first;});
    vector<int>out;for(auto q:clusters)if(q.first>=5&&q.first<=(int)y.size()-5)out.push_back(q.first);
    return out;
}

Model fitPiecewiseForcedCuts(const vector<ld>&x,const vector<ld>&y,const vector<int>&cuts){
    if(x.size()!=y.size()||cuts.empty()||cuts.size()>3)return{};
    const int n=sizeToInt(x.size());vector<int>c={0};for(int q:cuts)if(q>=5&&q<=n-5)c.push_back(q);c.push_back(n);sort(c.begin(),c.end());c.erase(unique(c.begin(),c.end()),c.end());if(c.size()<3)return{};
    struct Part{ld lo=0,hi=0;Model m;};vector<Part>parts;ld totalSSE=0;int params=0;
    for(size_t j=0;j+1<c.size();++j){int lo=c[j],hi=c[j+1];if(hi-lo<6)return{};vector<ld>tx(x.begin()+lo,x.begin()+hi),ty(y.begin()+lo,y.begin()+hi);Model m=fitBestLocalPieceModel(tx,ty);if(!m.valid||!m.predict)return{};for(size_t i=0;i<tx.size();++i){ld v=m.predict(tx[i]);if(!isfinite(v))return{};ld e=ty[i]-v;totalSSE+=e*e;}params+=max(1,m.complexity);parts.push_back({x[lo],x[hi-1],m});}
    Model out;out.valid=true;out.name="Adaptive piecewise function / calculus break model";out.category="Piecewise / segmented function model";out.complexity=params;
    out.predict=[parts](ld z){size_t k=0;while(k+1<parts.size()&&z>=parts[k+1].lo)++k;return parts[k].m.predict(z);};
    out.fitted.reserve(n);for(ld z:x)out.fitted.push_back(out.predict(z));out.nextValue=out.predict(x.back()+1);
    ostringstream os;os<<"piecewise function with "<<parts.size()<<" calculus-selected domains: ";for(size_t j=0;j<parts.size();++j){if(j)os<<" | ";os<<"["<<fmt(parts[j].lo,5)<<","<<fmt(parts[j].hi,5)<<"] -> "<<parts[j].m.name;}out.formula=os.str();
    out.note="Domain boundaries were selected from isolated jumps and/or left-versus-right derivative changes; local model families are chosen separately on each domain.";
    return out;
}


Analysis analyzeSequenceCore(const vector<ld>&x,const vector<ld>&y,int horizon,ld requestedTolerancePct=-1.0L){
    Analysis a;
    if(y.size()<3||x.size()!=y.size()||horizon<1)return a;
    a.tolerancePct=requestedTolerancePct>=0?requestedTolerancePct:adaptiveTolerancePct(y);
    a.changePoints=detectChangePoints(y);
    a.observedZeroCrossings=observedZeroCrossings(x,y);
    a.observedTurningPoints=observedTurningPoints(x,y);
    a.observedInflectionPoints=observedInflectionPoints(x,y);
    a.calculusSummary=summarizeObservedCalculus(x,y,a.observedZeroCrossings,a.observedTurningPoints,a.observedInflectionPoints);

    // Index-based sequence rules are only safe when x is the canonical unit grid.
    // Graph/curve inputs otherwise get judged in their physical x-domain.
    const bool unitGrid=isUnitSpacedGrid(x);
    vector<Model> exact;
    if(unitGrid){
        exact.push_back(exactAP(x,y));exact.push_back(exactGP(x,y));exact.push_back(exactFactorial(x,y));
        exact.push_back(exactFibonacci(x,y));exact.push_back(exactRecurrence2(x,y));exact.push_back(exactPowerFamily(x,y));
        exact.push_back(exactCatalan(x,y));exact.push_back(exactBell(x,y));exact.push_back(exactPrimeSequence(x,y));
        exact.push_back(exactPolygonal(x,y));exact.push_back(exactPeriodic(x,y));exact.push_back(exactEventualCycle(x,y));
        exact.push_back(exactInterleaved(x,y));exact.push_back(detectLCG(x,y));exact.push_back(detectXorShift32(x,y));
        exact.push_back(exactPermutation(y));exact.push_back(exactDistinctIntegerSet(y));
        exact.push_back(exactAffine(x,y));
        exact.push_back(exactStaircase(x,y));
        exact.push_back(fitRobustStaircaseProgression(x,y));
        exact.push_back(exactStepFunction(x,y));
    }else{
        exact.push_back(exactAffine(x,y));
        exact.push_back(exactStaircase(x,y));
        exact.push_back(exactStepFunction(x,y));
    }

    // Additional exact finite-difference detector for arbitrary low-degree
    // polynomial sequences (including shifted/scaled cases on the given x grid).
    if(y.size()>=6 && isUnitSpacedGrid(x)){
        vector<ld>cur=y;int degreeFound=-1;
        for(int d=1;d<=min(10,(int)y.size()-2);++d){
            vector<ld>next;for(size_t i=1;i<cur.size();++i)next.push_back(cur[i]-cur[i-1]);cur.swap(next);
            if(cur.size()<2)break;
            if(all_of(cur.begin()+1,cur.end(),[&](ld q){return approxEqual(q,cur.front(),1e-9L,1e-9L);})){degreeFound=d;break;}
        }
        if(degreeFound>=0){
            vector<ld> diffs;cur=y;diffs.push_back(y[0]);
            vector<ld> row=y;
            vector<vector<ld>> table;
            table.push_back(y);
            for(int d=1;d<=degreeFound;++d){vector<ld>nxt;for(size_t i=1;i<row.size();++i)nxt.push_back(row[i]-row[i-1]);table.push_back(nxt);row.swap(nxt);}
            Model m;m.valid=m.exact=true;m.name="Finite-difference polynomial degree "+to_string(degreeFound);
            m.category="Exact finite-difference polynomial";m.complexity=degreeFound+1;
            const ld x0=x.front();
            vector<ld>fd0;fd0.reserve((size_t)degreeFound+1);
            for(int k=0;k<=degreeFound;++k)fd0.push_back(table[k].front());
            // Convert Newton-forward basis sum d_k * C(t,k), t=x-x0, to the
            // ordinary power basis.  This gives a genuinely continuous curve
            // for forecasting and lets the numerical root/calculus layer use
            // fast polynomial root localization rather than a grid scan.
            vector<ld>polyT(1,0);
            vector<ld>basis(1,1);
            ld invFact=1;
            for(int k=0;k<=degreeFound;++k){
                if(k>0){
                    vector<ld>nb(basis.size()+1,0);
                    for(size_t j=0;j<basis.size();++j){nb[j]-=(ld)(k-1)*basis[j];nb[j+1]+=basis[j];}
                    basis.swap(nb);
                    invFact/=k;
                }
                if(polyT.size()<basis.size())polyT.resize(basis.size(),0);
                for(size_t j=0;j<basis.size();++j)polyT[j]+=fd0[k]*invFact*basis[j];
            }
            vector<ld>polyX=polyShiftVariable(polyT,x0);
            auto rootsX=polynomialRealRoots(polyX);
            auto dX=polyDerivativeCoeffs(polyX);
            auto d2X=polyDerivativeCoeffs(dX);
            m.approxRealRoots=rootsX;
            m.approxDerivativeRoots=polynomialRealRoots(dX);
            m.approxInflectionRoots=polynomialRealRoots(d2X);
            m.calculusSummary="Polynomial calculus: roots from fitted polynomial coefficients; derivative roots are extrema candidates; second-derivative roots are inflection candidates.";
            m.predict=[polyT,x0](ld z){
                const ld t=z-x0;
                ld val=0;
                for(auto it=polyT.rbegin();it!=polyT.rend();++it)val=val*t+*it;
                return val;
            };
            m.fitted=y;m.nextValue=m.predict(x.back()+1);
            m.formula="Polynomial sequence from constant finite differences, degree="+to_string(degreeFound);
            exact.push_back(m);
        }
    }

    for(auto&m:exact){
        if(!m.valid)continue;
        if(m.name.find("Permutation")!=string::npos||m.name.find("Distinct integer")!=string::npos){
            a.found=true;a.deterministicNext=false;a.category=m.category;a.logic="Exact set/order structure detected; membership alone does not determine the next term.";a.formula=m.formula;a.evidenceScore=100;a.model=m;a.warning=m.note;return a;
        }
        auto preds=forecastFromModel(m,x.back(),horizon);
        if(preds.size()!=static_cast<size_t>(horizon))continue;
        a.found=true;a.deterministicNext=true;a.category=m.category;a.logic="Exact algorithmic rule validated against every supplied observation.";a.formula=m.formula;a.evidenceScore=100;a.model=m;a.predictions=preds;a.nextValue=preds.front();attachNumericRoots(a.model,x,true);return a;
    }

    // Calculus-driven piecewise-linear regime path.  This comes before global
    // polynomial/model selection so control-system style ramps, holds, falls,
    // lifts and domain changes are represented as separate local rules rather than
    // being misclassified as one smooth law.
    {
        Model regime=fitPiecewiseLinearRegimeModel(x,y);
        if(regime.valid&&regime.predict){
            auto [rrms,rmae]=errorMetrics(y,regime.fitted);
            ld scale=max<ld>(robustScale(y),1e-9L);
            auto rp=forecastFromModel(regime,x.back(),horizon);
            if(rrms/scale<0.18L && rp.size()==static_cast<size_t>(horizon) && !absurdPrediction(regime,y,rp)){
                regime.window=sizeToInt(y.size());
                regime.metrics.rmse=rrms;regime.metrics.mae=rmae;regime.metrics.r2=r2Score(y,regime.fitted);
                regime.metrics.normalizedRMSE=rrms/scale;regime.metrics.score=clamp01(1.0L/(1.0L+rrms/scale))*82.0L;
                regime.nextValue=rp.front();
                a.model=regime;a.predictions=rp;a.nextValue=rp.front();a.found=true;a.deterministicNext=false;a.shapeBased=true;
                a.category=regime.name;a.formula=regime.formula;
                a.logic="Derivative-segmented calculus route: local first-derivative regimes, flat holds and slope changes were detected as domain boundaries; each domain was fitted separately and the final domain was extrapolated.";
                a.warning="The domain segmentation is evidence-based. A future regime change can only be predicted when its timing/pattern is supported by repeated observed regimes.";
                attachNumericRoots(a.model,x,true);return a;
            }
        }
    }

    // Fast algebraic curve path: when a low-degree polynomial explains the entire
    // supplied curve to numerical precision, prefer it before expensive rolling
    // model selection. This is especially useful for multi-root curves.
    if(!unitGrid && y.size()>=8){
        Model bestPoly; ld bestRel=numeric_limits<ld>::infinity();
        for(int deg=1;deg<=6;++deg){
            Model pm=fitPolyModel(x,y,deg);if(!pm.valid||!pm.predict)continue;
            auto [rms,mae]=errorMetrics(y,pm.fitted);ld rel=rms/max<ld>(robustScale(y),1e-9L);
            if(rel<bestRel){bestRel=rel;bestPoly=pm;bestPoly.metrics.rmse=rms;bestPoly.metrics.mae=mae;}
        }
        if(bestPoly.valid&&bestRel<1e-8L){
            a.model=bestPoly;a.found=true;a.deterministicNext=true;a.category=bestPoly.category;a.logic="Low-degree polynomial relation validated to numerical precision; bypassed generic model search.";a.formula=bestPoly.formula;a.predictions=forecastFromModel(bestPoly,x.back(),horizon);a.nextValue=a.predictions.empty()?bestPoly.nextValue:a.predictions.front();attachNumericRoots(a.model,x,true);return a;
        }
    }

    // Strong domain-boundary route: a large isolated jump or a clear left/right
    // derivative mismatch is evidence for a piecewise/discontinuous function.
    auto strongBreaks=strongShapeBreakCandidates(x,y);
    if(!strongBreaks.empty()&&strongBreaks.size()<=3){
        Model pw=fitPiecewiseForcedCuts(x,y,strongBreaks);
        if(!pw.valid)pw=fitPiecewiseAdaptiveModel(x,y,2);
        if(pw.valid&&pw.predict){auto pp=forecastFromModel(pw,x.back(),horizon);if(pp.size()==static_cast<size_t>(horizon)&&!absurdPrediction(pw,y,pp)){
            pw.window=sizeToInt(y.size());a.model=pw;a.predictions=pp;a.nextValue=pp.front();a.found=true;a.deterministicNext=false;a.shapeBased=true;a.category=pw.name;a.formula=pw.formula;
            a.logic="Piecewise route: isolated discontinuities and/or statistically strong left/right derivative changes were detected; the domain was segmented and each segment received its own local model.";
            a.warning="The segmentation is evidence-based rather than a proof of the underlying generating function; the final segment is extrapolated forward.";
            attachNumericRoots(a.model,x,true);return a;}}
    }

    // First try a nonstationary oscillation model when extrema spacing itself changes.
    if(a.observedTurningPoints.size()>=7){
        Model chirp=fitAdaptiveChirpEnvelopeModel(x,y);
        if(chirp.valid&&chirp.predict){
            auto [crms,cmae]=errorMetrics(y,chirp.fitted);ld scale=max<ld>(robustScale(y),1e-9L);
            auto cp=forecastFromModel(chirp,x.back(),horizon);
            if(isfinite(crms)&&crms/scale<0.35L&&cp.size()==static_cast<size_t>(horizon)&&!absurdPrediction(chirp,y,cp)){
                chirp.window=sizeToInt(y.size());chirp.metrics.rmse=crms;chirp.metrics.mae=cmae;chirp.metrics.r2=r2Score(y,chirp.fitted);chirp.metrics.normalizedRMSE=crms/scale;chirp.metrics.score=clamp01(1.0L/(1.0L+crms/scale))*75.0L;chirp.nextValue=cp.front();
                a.model=chirp;a.predictions=cp;a.nextValue=cp.front();a.found=true;a.deterministicNext=false;a.shapeBased=true;a.category=chirp.name;a.formula=chirp.formula;
                a.logic="Adaptive calculus route: turning-point spacing was used to estimate frequency drift, while numerical derivatives and curvature constrained an evolving-period oscillatory model.";
                a.warning="This is shape-guided rather than deterministic; the finite sample supports changing-period oscillation but cannot uniquely determine arbitrary future behavior.";
                attachNumericRoots(a.model,x,true);return a;
            }
        }
    }

    // Route clearly oscillatory trajectories to the calculus/shape engine before
    // the large general candidate sweep. This avoids forcing a smooth, repeated
    // extrema pattern through a straight-line winner merely because early folds
    // are harder than the latest cycle. It also dramatically reduces runtime on
    // long oscillatory curves.
    if(a.observedTurningPoints.size()>=5){
        ld ep=estimateExtremaPeriod(x,y);
        Model shape=fitOscillatoryEnvelopeModel(x,y,ep);
        if(shape.valid&&shape.predict){
            auto [srms,smae]=errorMetrics(y,shape.fitted);ld scale=max<ld>(robustScale(y),1e-9L);
            if(isfinite(srms)&&srms/scale<0.45L){
                auto sp=forecastFromModel(shape,x.back(),horizon);
                if(sp.size()==static_cast<size_t>(horizon)&&!absurdPrediction(shape,y,sp)){
                    shape.window=sizeToInt(y.size());shape.metrics.rmse=srms;shape.metrics.mae=smae;shape.metrics.r2=r2Score(y,shape.fitted);shape.metrics.normalizedRMSE=srms/scale;shape.metrics.score=clamp01(1.0L/(1.0L+srms/scale))*70.0L;shape.nextValue=sp.front();
                    a.model=shape;a.predictions=sp;a.nextValue=sp.front();a.found=true;a.deterministicNext=false;a.shapeBased=true;a.category=shape.name;a.formula=shape.formula;
                    a.logic="Primary calculus/shape route: numerical roots, extrema, cycle spacing, amplitude evolution and curvature were strong enough to justify a local oscillatory extrapolation.";
                    a.warning="This is shape-guided rather than deterministic; the data supports a repeated oscillatory trajectory, but finite samples cannot prove the unique future function.";
                    attachNumericRoots(a.model,x,true);return a;
                }
            }
        }
    }

    vector<FamilySpec>specs=makeSpecs(x,y);
    vector<Model>scored;
    const int n=sizeToInt(y.size());
    vector<int>windows=candidateWindows(n);

    auto familyExpensive=[](const string&f){
        return f=="single_feature"||f=="symbolic"||f=="rbf"||f=="chirp"||f=="damped_sine"||f=="root_poly"||f=="segmented"||f=="piecewise_adaptive"||f=="seasonal"||f=="periodic_delta"||f=="osc_envelope"||f=="adaptive_chirp_envelope"||f=="huber_basis"||f=="elastic_basis"||f=="lasso_basis";
    };
    for(const auto&spec:specs){
        vector<int>evalWindows=windows;
        if(familyExpensive(spec.family) && evalWindows.size()>4){
            vector<int>reduced;
            int nn=sizeToInt(y.size());
            for(int q:{nn,(int)llround(.70L*nn),(int)llround(.50L*nn),(int)llround(.30L*nn)})
                if(q>=8&&q<=nn) reduced.push_back(q);
            sort(reduced.begin(),reduced.end());reduced.erase(unique(reduced.begin(),reduced.end()),reduced.end());
            if(!reduced.empty()) evalWindows.swap(reduced);
        }
        for(int w:evalWindows){
            if(w<8||w>n-1)continue;
            ChunkValidation cv=validateRolling(x,y,spec,w,a.tolerancePct,requestedTolerancePct>=0);
            if(!cv.ok)continue;
            vector<ld>tx(x.end()-w,x.end()),ty(y.end()-w,y.end());
            Model fm=fitByFamily(spec.family,tx,ty,spec.degree,spec.param,spec.param2);
            if(!fm.valid||!fm.predict)continue;
            fm.window=w;fm.metrics=cv.metrics;fm.validationResiduals=cv.residuals;
            vector<ld>pred=forecastFromModel(fm,x.back(),horizon);
            if(pred.size()!=static_cast<size_t>(horizon))continue;
            if(absurdPrediction(fm,y,pred))continue;
            fm.nextValue=pred.front();fm.fitted=finiteFitted(fm,x,y);
            scored.push_back(fm);
        }
    }

    if(scored.empty()){
        a.found=false;a.deterministicNext=false;a.category="No validated predictive pattern";
        a.logic="All exact rules and machine-learning/statistical candidates were rejected by rolling unseen-data validation or stability checks.";
        a.warning="Try a longer sequence, inspect diagnostics, or provide a model/domain constraint. No finite method can guarantee perfect extrapolation from insufficient data.";
        return a;
    }

    sort(scored.begin(),scored.end(),[](const Model&A,const Model&B){
        if(fabsl(A.metrics.score-B.metrics.score)>1e-10L)return A.metrics.score>B.metrics.score;
        if(A.metrics.p90DeviationPct!=B.metrics.p90DeviationPct)return A.metrics.p90DeviationPct<B.metrics.p90DeviationPct;
        if(A.complexity!=B.complexity)return A.complexity<B.complexity;
        return A.name<B.name;
    });

    // Consensus ensemble: use only top independently rolling-validated models,
    // and only when their forecasts are reasonably close. This avoids blindly
    // averaging wildly different extrapolations.
    if(scored.size()>=3){
        vector<Model>members;
        vector<ld>weights;
        const ld topScore=scored.front().metrics.score;
        set<string>cats;
        for(auto&m:scored){
            if((int)members.size()>=5)break;
            if(m.metrics.score+8.0L<topScore)break;
            if(cats.count(m.name.substr(0,m.name.find(' '))))continue;
            members.push_back(m);
            weights.push_back(max<ld>(0.05L,m.metrics.score*m.metrics.score));
            cats.insert(m.name.substr(0,m.name.find(' ')));
        }
        if(members.size()>=3){
            vector<ld>ens(horizon,0);
            ld ws=accumulate(weights.begin(),weights.end(),0.0L);
            for(size_t j=0;j<members.size();++j){
                auto p=forecastFromModel(members[j],x.back(),horizon);
                if(p.size()!=static_cast<size_t>(horizon)){ens.clear();break;}
                for(int h=0;h<horizon;++h)ens[h]+=weights[j]*p[h]/ws;
            }
            if(ens.size()==static_cast<size_t>(horizon)){
                ld scale=max({robustScale(y),1.0L,fabsl(quantile(y,.5L))});
                ld disagree=0;
                for(int h=0;h<horizon;++h){
                    ld mn=members.front().predict(x.back()+h+1),mx=mn;
                    for(auto&mm:members){ld q=mm.predict(x.back()+h+1);mn=min(mn,q);mx=max(mx,q);}
                    disagree=max(disagree,(mx-mn)/max(scale,1e-9L));
                }
                if(disagree<max<ld>(0.05L,a.tolerancePct/100.0L*4.0L+0.05L)){
                    Model em;
                    em.valid=true;em.name="Consensus ensemble (top rolling-validated models)";em.category="Ensemble machine learning";em.complexity=sizeToInt(members.size());
                    em.predict=[members,weights,ws](ld z){ld s=0;for(size_t i=0;i<members.size();++i)s+=weights[i]*members[i].predict(z)/ws;return s;};
                    em.nextValue=ens.front();em.fitted.reserve(x.size());for(ld xx:x)em.fitted.push_back(em.predict(xx));
                    em.metrics=scored.front().metrics;
                    ld avgScore=0;for(const auto&m:members)avgScore+=m.metrics.score;avgScore/=members.size();em.metrics.score=avgScore;
                    em.window=scored.front().window;
                    em.formula="Weighted consensus of "+to_string(members.size())+" independently rolling-validated models";
                    em.note="Consensus is a post-validation ensemble; component models were independently validated. The ensemble score is an evidence summary, not a probability.";
                    scored.push_back(em);
                    sort(scored.begin(),scored.end(),[](const Model&A,const Model&B){return A.metrics.score>B.metrics.score;});
                }
            }
        }
    }

    a.leaderboard=scored;
    a.model=scored.front();
    a.evidenceScore=a.model.metrics.score;
    bool reliable=a.model.metrics.stable;
    if(!reliable){
        // Second-stage shape/dynamics fallback.  This is intentionally separate
        // from the ordinary reliability gate: a smooth oscillatory trajectory can
        // have poor early rolling folds while its latest extrema/period/amplitude
        // still contain useful local information.  Never label this deterministic.
        if(a.observedTurningPoints.size()>=5){
            ld p=estimateExtremaPeriod(x,y);
            Model shape=fitOscillatoryEnvelopeModel(x,y,p);
            if(shape.valid&&shape.predict){
                vector<ld>sp=forecastFromModel(shape,x.back(),horizon);
                if(sp.size()==static_cast<size_t>(horizon)&&!absurdPrediction(shape,y,sp)){
                    shape.window=sizeToInt(y.size());shape.nextValue=sp.front();shape.metrics=a.model.metrics;
                    shape.metrics.stable=false;shape.metrics.rollingValidated=false;
                    shape.metrics.score=max<ld>(shape.metrics.score,45.0L);
                    a.model=shape;a.predictions=sp;a.nextValue=sp.front();a.found=true;a.deterministicNext=false;a.shapeBased=true;a.category=shape.name;
                    a.logic="Shape-aware calculus fallback: numerical zero crossings, turning points, inferred cycle length, amplitude evolution and local curvature were used after no global model cleared the rolling reliability gate.";
                    a.formula=shape.formula;a.warning="No single global model passed the strict rolling-stability gate. This forecast is a shape-guided heuristic, not a guaranteed continuation.";
                    attachNumericRoots(a.model,x,true);
                    auto half=empiricalForecastBand(a.predictions,a.model.validationResiduals,0.90L);
                    if(half.size()==a.predictions.size()){a.predictionLow.resize(horizon);a.predictionHigh.resize(horizon);for(int i=0;i<horizon;++i){a.predictionLow[i]=a.predictions[i]-half[i];a.predictionHigh[i]=a.predictions[i]+half[i];}}
                    return a;
                }
            }
        }
        a.found=false;a.deterministicNext=false;a.category="No stable predictive pattern";a.logic="Candidates were compared with moving rolling-origin validation, and the shape-aware fallback also lacked enough structural evidence for a responsible forecast.";a.formula=a.model.formula;
        a.warning="Best validated candidate: "+a.model.name+"; p90 deviation="+fmt(a.model.metrics.p90DeviationPct,3)+"%, tolerance="+fmt(a.tolerancePct,3)+"%.";
        return a;
    }

    a.found=true;a.deterministicNext=true;a.category=a.model.name;
    a.logic="Exact rules -> nonlinear/symbolic/ML candidates -> moving training windows -> rolling-origin unseen-data validation -> stability gate -> consensus/uncertainty diagnostics.";
    attachNumericRoots(a.model,x,true);
    a.formula=a.model.formula;a.predictions=forecastFromModel(a.model,x.back(),horizon);a.nextValue=a.predictions.empty()?a.model.nextValue:a.predictions.front();
    auto half=empiricalForecastBand(a.predictions,a.model.validationResiduals,0.90L);
    if(half.size()==a.predictions.size()){
        a.predictionLow.resize(horizon);a.predictionHigh.resize(horizon);
        for(int i=0;i<horizon;++i){a.predictionLow[i]=a.predictions[i]-half[i];a.predictionHigh[i]=a.predictions[i]+half[i];}
    }
    return a;
}

// Re-expresses an Analysis that was actually computed on the sign-mirrored
// series (-y) back in terms of the original series, by negating every
// number it produced (predictions, band, fitted values, and each callable
// model's predict()). Used by the mirror-retry wrapper below.
Analysis negateAnalysisForMirror(Analysis a) {
    auto negatePredictFn = [](function<ld(ld)> f) {
        return function<ld(ld)>([f](ld z){ ld v=f(z); return isfinite(v) ? -v : v; });
    };
    if (a.model.valid && a.model.predict) {
        a.model.predict = negatePredictFn(a.model.predict);
        if (isfinite(a.model.nextValue)) a.model.nextValue = -a.model.nextValue;
        for (auto& f : a.model.fitted) f = -f;
        a.model.formula = "-1 * [ " + a.model.formula + " ]  (rule found on the sign-mirrored series -y)";
        a.model.name += " (sign-mirrored)";
    }
    for (auto& m : a.leaderboard) {
        if (!m.predict) continue;
        m.predict = negatePredictFn(m.predict);
        for (auto& f : m.fitted) f = -f;
        m.name += " (mirrored)";
    }
    for (auto& p : a.predictions) p = -p;
    if (isfinite(a.nextValue)) a.nextValue = -a.nextValue;
    if (a.predictionLow.size() == a.predictionHigh.size()) {
        for (size_t i = 0; i < a.predictionLow.size(); ++i) {
            ld lo = -a.predictionHigh[i], hi = -a.predictionLow[i];
            a.predictionLow[i] = lo; a.predictionHigh[i] = hi;
        }
    }
    a.logic += " The winning rule was actually found on the sign-mirrored series (-y) and is reported back negated; "
               "this catches patterns (e.g. reflected/alternating shapes) that only reveal themselves once the sign is flipped.";
    return a;
}

// Public entry point. Tries the series as given, and -- unless the direct
// fit is already an exact rule -- also tries the sign-mirrored series (-y)
// and keeps whichever produced the more convincing, validated fit. This is
// a cheap, general-purpose symmetry probe: some patterns (reflected growth,
// sign-alternating dynamics, |x|-shaped behaviour observed from the
// underside) are much easier to validate on -y than on y directly, even
// though the two are mathematically equivalent up to a sign flip.
Analysis analyzeSequence(const vector<ld>&x,const vector<ld>&y,int horizon,ld requestedTolerancePct=-1.0L,bool allowMirror=true){
    Analysis direct = analyzeSequenceCore(x,y,horizon,requestedTolerancePct);
    // The mirror pass reruns the entire search, so for performance it is only
    // worth paying for when the direct fit is not already strong -- an
    // already-confident fit is very unlikely to be beaten by its own mirror.
    if (!allowMirror || y.size()<3 || (direct.found && (direct.model.exact || direct.evidenceScore>=80.0L))) return direct;
    vector<ld> negY; negY.reserve(y.size()); for (ld v: y) negY.push_back(-v);
    Analysis mirrored = analyzeSequenceCore(x,negY,horizon,requestedTolerancePct);
    bool mirrorIsBetter = mirrored.found && (!direct.found || mirrored.evidenceScore > direct.evidenceScore + 3.0L);
    if (mirrorIsBetter) return negateAnalysisForMirror(mirrored);
    return direct;
}

// ----------------------------- Multivariate joint models ------------
struct MultiModel {
    bool valid=false;
    string name, formula;
    int lag=1;
    int variables=0;
    ld lambda=0;
    Metrics metrics;
    vector<vector<ld>> coef; // response x [intercept + lags*variables]
    vector<vector<ld>> history;
    function<vector<vector<ld>>(const vector<vector<ld>>&,int)> forecast;
};

vector<vector<ld>> makeVARDesign(const vector<vector<ld>>& rows,int lag,vector<vector<ld>>& targets){
    const int n=sizeToInt(rows.size()),k=rows.empty()?0:sizeToInt(rows[0].size());
    vector<vector<ld>>A;targets.clear();
    if(n<=lag)return A;
    for(int t=lag;t<n;++t){
        vector<ld>row(1,1.0L);
        for(int l=1;l<=lag;++l)for(int j=0;j<k;++j)row.push_back(rows[t-l][j]);
        A.push_back(row);targets.push_back(rows[t]);
    }
    return A;
}

bool fitVAR(const vector<vector<ld>>& rows,int lag,ld lambda,MultiModel& out){
    if(rows.empty()||lag<1||rows.size()<static_cast<size_t>(lag+8))return false;
    const int k=sizeToInt(rows[0].size());
    vector<vector<ld>>targets;
    auto A=makeVARDesign(rows,lag,targets);
    if(A.empty())return false;
    out.coef.assign(k,{});
    for(int j=0;j<k;++j){
        vector<ld>yy;yy.reserve(targets.size());for(auto&r:targets)yy.push_back(r[j]);
        if(!ridgeSolve(A,yy,lambda,out.coef[j],false))return false;
    }
    out.valid=true;out.lag=lag;out.variables=k;out.lambda=lambda;out.history=rows;
    out.name="Joint VAR("+to_string(lag)+") ridge";
    out.formula="y_t = B0 + B1*y_(t-1) + ... + B"+to_string(lag)+"*y_(t-"+to_string(lag)+")";
    out.forecast=[coef=out.coef,lag,k](const vector<vector<ld>>& hist,int h){
        vector<vector<ld>>work=hist,fut;
        if(work.empty())return fut;
        for(int step=0;step<h;++step){
            vector<ld>row(1,1.0L);
            for(int l=1;l<=lag;++l){
                const auto&src=work[work.size()-l];
                for(int j=0;j<k;++j)row.push_back(src[j]);
            }
            vector<ld>next(k,0);
            for(int j=0;j<k;++j){
                for(size_t q=0;q<row.size()&&q<coef[j].size();++q)next[j]+=coef[j][q]*row[q];
            }
            if(any_of(next.begin(),next.end(),[](ld v){return !isfinite(v)||fabsl(v)>1e300L;}))return vector<vector<ld>>{};
            work.push_back(next);fut.push_back(next);
        }
        return fut;
    };
    return true;
}

ld multivariateScale(const vector<vector<ld>>&rows){
    ld s=1;
    for(const auto&r:rows)for(ld v:r)s=max(s,fabsl(v));
    return s;
}

bool validateVARRolling(const vector<vector<ld>>& rows,int lag,ld lambda,ld tol,bool strict,MultiModel& finalModel){
    const int n=sizeToInt(rows.size()),k=sizeToInt(rows[0].size());
    if(n<lag+12)return false;
    vector<Metrics>blocks;vector<ld>allA,allP,allD;
    vector<ld>scales(k,1.0L);
    for(int j=0;j<k;++j){vector<ld>col;col.reserve(rows.size());for(const auto&r:rows)col.push_back(r[j]);scales[j]=max<ld>(robustScale(col),1e-12L);}
    int window=min(n,max(lag+8,(int)llround(.65L*n)));vector<int>horizons={1,2,max(3,n/12)};
    for(int h:horizons){if(window+h>n)continue;auto origins=selectValidationOrigins(window,n-h,24);for(ld oo:origins){int origin=(int)oo,start=max(0,origin-window);vector<vector<ld>>train(rows.begin()+start,rows.begin()+origin);MultiModel mm;if(!fitVAR(train,lag,lambda,mm))continue;auto pred=mm.forecast(train,h);if(pred.size()!=static_cast<size_t>(h))continue;vector<ld>a,p,dv;bool bad=false;for(int i=0;i<h&&!bad;++i)for(int j=0;j<k;++j){ld av=rows[origin+i][j],pv=pred[i][j];if(!isfinite(pv)||fabsl(pv)>1e300L){bad=true;break;}a.push_back(av);p.push_back(pv);dv.push_back(relativeDeviationPct(av,pv,scales[j]));}if(bad||a.empty())continue;auto[rme,mae]=errorMetrics(a,p);Metrics b;b.nTrain=train.size();b.nValid=a.size();b.rmse=rme;b.mae=mae;b.r2=r2Score(a,p);b.meanDeviationPct=meanOf(dv);b.medianDeviationPct=quantile(dv,.5L);b.maxDeviationPct=*max_element(dv.begin(),dv.end());b.p90DeviationPct=quantile(dv,.9L);b.hitRate=count_if(dv.begin(),dv.end(),[&](ld q){return q<=tol;})/(ld)dv.size();b.smapePct=smapePct(a,p);b.validationBlocks=1;bool pass=strict?(b.p90DeviationPct<=tol&&b.maxDeviationPct<=tol*1.25L):(b.medianDeviationPct<=tol&&b.p90DeviationPct<=2*tol);b.blockPassRate=pass?1.0L:0.0L;blocks.push_back(b);allA.insert(allA.end(),a.begin(),a.end());allP.insert(allP.end(),p.begin(),p.end());allD.insert(allD.end(),dv.begin(),dv.end());}
    }
    if(blocks.empty())return false;
    (void)finalModel;
    int total=0,passed=0;
    ld wr=0,wm=0,wr2=0,wh=0,wsm=0,sumAbs=0,sumErr=0;
    for(size_t i=0;i<allA.size();++i){sumAbs+=fabsl(allA[i]);sumErr+=fabsl(allA[i]-allP[i]);}
    for(auto&b:blocks){int w=max(1,b.nValid);total+=w;wr+=b.rmse*w;wm+=b.mae*w;wr2+=b.r2*w;wh+=b.hitRate*w;wsm+=b.smapePct*w;if(b.blockPassRate>0.5L)++passed;}
    Metrics m;m.nTrain=blocks.front().nTrain;m.nValid=total;m.validationBlocks=blocks.size();m.rmse=wr/total;m.mae=wm/total;m.r2=wr2/total;m.hitRate=wh/total;m.smapePct=wsm/total;m.meanDeviationPct=meanOf(allD);m.medianDeviationPct=quantile(allD,.5L);m.p90DeviationPct=quantile(allD,.9L);m.maxDeviationPct=*max_element(allD.begin(),allD.end());m.wapePct=100.0L*sumErr/max<ld>(sumAbs,1e-12L);m.blockPassRate=(ld)passed/blocks.size();m.normalizedRMSE=m.rmse; m.rollingValidated=true;
    m.stable=strict?(m.p90DeviationPct<=tol&&m.blockPassRate>=.70L):(m.medianDeviationPct<=tol&&m.p90DeviationPct<=2*tol&&m.blockPassRate>=.60L);
    m.scoreRMSE=1/(1+m.normalizedRMSE);m.scoreMAE=1/(1+m.wapePct/100);m.scoreP90=expl(-m.p90DeviationPct/max<ld>(tol,1e-6L));m.scoreR2=clamp01(m.r2);m.scoreDirection=.5L;m.scoreCorrelation=.5L;m.scoreStability=m.blockPassRate;m.scoreComplexity=1/sqrtl(1+.03L*max(1,lag));m.score=100*powl(max<ld>(1e-18L,m.scoreRMSE*m.scoreMAE*m.scoreP90*max<ld>(1e-6L,m.scoreR2)*m.scoreStability*m.scoreComplexity),1.0L/6.0L);
    finalModel.metrics=m;return true;
}

RelationAnalysis analyzeRelation(const vector<ld>&x,const vector<ld>&y,const vector<ld>&futureX,int horizon,ld requestedTolerancePct=-1.0L){
    RelationAnalysis a;
    if(x.size()!=y.size()||x.size()<8||horizon<1)return a;
    a.tolerancePct=requestedTolerancePct>=0?requestedTolerancePct:adaptiveTolerancePct(y);
    vector<FamilySpec>specs=makeSpecs(x,y);vector<Model>scored;vector<int>windows=candidateWindows(sizeToInt(y.size()));
    auto familyExpensive=[](const string&f){
        return f=="single_feature"||f=="symbolic"||f=="rbf"||f=="chirp"||f=="damped_sine"||f=="root_poly"||f=="segmented"||f=="piecewise_adaptive"||f=="seasonal"||f=="periodic_delta"||f=="huber_basis"||f=="elastic_basis"||f=="lasso_basis";
    };
    for(const auto&spec:specs){
        vector<int>evalWindows=windows;
        if(familyExpensive(spec.family) && evalWindows.size()>4){
            vector<int>reduced;
            int nn=sizeToInt(y.size());
            for(int q:{nn,(int)llround(.70L*nn),(int)llround(.50L*nn),(int)llround(.30L*nn)})
                if(q>=8&&q<=nn) reduced.push_back(q);
            sort(reduced.begin(),reduced.end());reduced.erase(unique(reduced.begin(),reduced.end()),reduced.end());
            if(!reduced.empty()) evalWindows.swap(reduced);
        }
        for(int w:evalWindows){
            if(w>sizeToInt(y.size())-1)continue;
            auto cv=validateRolling(x,y,spec,w,a.tolerancePct,requestedTolerancePct>=0);
            if(!cv.ok)continue;
            vector<ld>tx(x.end()-w,x.end()),ty(y.end()-w,y.end());
            Model m=fitByFamily(spec.family,tx,ty,spec.degree,spec.param,spec.param2);if(!m.valid)continue;
            m.window=w;m.metrics=cv.metrics;
            vector<ld>fp;bool ok=true;for(ld fx:futureX){ld q=m.predict(fx);if(!isfinite(q)||fabsl(q)>1e300L){ok=false;break;}fp.push_back(q);}
            if(!ok||fp.size()!=futureX.size()||absurdPrediction(m,y,fp))continue;
            scored.push_back(m);
        }
    }
    if(scored.empty()){a.found=false;a.category="No stable relation";a.logic="No candidate relation survived moving-window unseen-data validation.";a.warning="Relation rejected for this variable pair.";return a;}
    sort(scored.begin(),scored.end(),[](const Model&A,const Model&B){if(A.metrics.score!=B.metrics.score)return A.metrics.score>B.metrics.score;return A.metrics.p90DeviationPct<B.metrics.p90DeviationPct;});
    a.leaderboard=scored;a.model=scored.front();a.evidenceScore=a.model.metrics.score;a.formula=a.model.formula;
    bool ok=a.model.metrics.stable;
    if(!ok){a.found=false;a.category="Relation rejected";a.logic="The best relation missed the rolling deviation gate.";a.warning="p90 validation deviation="+fmt(a.model.metrics.p90DeviationPct,3)+"%, tolerance="+fmt(a.tolerancePct,3)+"%.";return a;}
    a.found=true;a.category=a.model.name;a.logic="Validated dependency of the current variable on the previously accepted variable.";for(ld fx:futureX)a.futurePredictions.push_back(a.model.predict(fx));return a;
}

MultiAnalysis analyzeMultivariate(const vector<vector<ld>>&rows,int horizon,ld requestedTolerancePct=-1.0L){
    MultiAnalysis out;out.rows=rows.size();out.variables=rows.empty()?0:rows[0].size();
    if(rows.size()<10||out.variables<2||horizon<1){out.failureReason="Need at least 10 rows and 2 variables.";return out;}
    for(const auto&r:rows)if((int)r.size()!=out.variables){out.failureReason="Rows have inconsistent variable counts.";return out;}
    out.validStructure=true;
    vector<ld>driver;driver.reserve(rows.size());for(const auto&r:rows)driver.push_back(r[0]);
    out.tolerancePct=requestedTolerancePct>=0?requestedTolerancePct:adaptiveTolerancePct(driver);

    // Evaluate joint VAR ridge models with rolling origins.
    vector<MultiModel> candidates;
    for(int lag=1;lag<=min(12,(int)rows.size()/4);++lag){
        for(ld lambda:{1e-4L,1e-3L,1e-2L,1e-1L,1.0L}){
            MultiModel mm;
            if(!fitVAR(rows,lag,lambda,mm))continue;
            MultiModel copy=mm;
            if(validateVARRolling(rows,lag,lambda,out.tolerancePct,requestedTolerancePct>=0,copy))candidates.push_back(copy);
        }
    }

    vector<ld>idx(rows.size());iota(idx.begin(),idx.end(),1);
    out.driver=analyzeSequence(idx,driver,horizon,requestedTolerancePct);

    if(!candidates.empty()){
        sort(candidates.begin(),candidates.end(),[](const MultiModel&A,const MultiModel&B){
            if(A.metrics.score!=B.metrics.score)return A.metrics.score>B.metrics.score;
            return A.lag<B.lag;
        });
        MultiModel best=candidates.front();
        auto fut=best.forecast(rows,horizon);
        if(fut.size()==static_cast<size_t>(horizon)){
            out.fullForecastAvailable=true;out.predictions=fut;
            // Present the joint model as a single stage so the output explicitly
            // distinguishes cross-variable learning from the older chain.
            RelationAnalysis joint;
            joint.found=true;joint.category=best.name;joint.logic="Joint multivariate ridge model learns every variable from lagged values of every variable simultaneously.";
            joint.formula=best.formula;joint.evidenceScore=best.metrics.score;joint.tolerancePct=out.tolerancePct;joint.model.valid=true;joint.model.name=best.name;joint.model.formula=best.formula;joint.model.category="Joint multivariate machine learning";joint.model.metrics=best.metrics;joint.futurePredictions= fut.front();
            out.links={joint};
            out.failureReason.clear();
            return out;
        }
    }

    // Fallback: preserve the original sequential dependency path, but now the
    // individual relation engine uses the same stronger rolling-validation set.
    if(!out.driver.found||!out.driver.deterministicNext){
        out.failureReason="Neither the joint VAR model nor the left-most univariate driver established a validated future path.";
        return out;
    }
    vector<ld>prevObs=driver,prevFuture=out.driver.predictions;
    out.predictions.assign(horizon,vector<ld>(out.variables,numeric_limits<ld>::quiet_NaN()));
    for(int h=0;h<horizon;++h)out.predictions[h][0]=prevFuture[h];
    out.links.clear();
    for(int j=1;j<out.variables;++j){
        vector<ld>target;for(auto&r:rows)target.push_back(r[j]);
        auto rel=analyzeRelation(prevObs,target,prevFuture,horizon,requestedTolerancePct);
        out.links.push_back(rel);
        if(!rel.found){out.failureReason="Joint model unavailable and chained relation rejected at variable "+to_string(j+1)+".";out.fullForecastAvailable=false;return out;}
        for(int h=0;h<horizon;++h)out.predictions[h][j]=rel.futurePredictions[h];
        prevObs=target;prevFuture=rel.futurePredictions;
    }
    out.fullForecastAvailable=true;return out;
}

// ----------------------------- Statistics ------------------------
ld normalSF(ld z){return 0.5L*erfcl(z/sqrtl(2.0L));}
ld chiSquarePApprox(ld chi,int df){if(df<=0||chi<0)return numeric_limits<ld>::quiet_NaN();ld a=pow(max<ld>(chi,0)/df,1.0L/3.0L);ld mu=1-2.0L/(9.0L*df),sd=sqrtl(2.0L/(9.0L*df));return normalSF((a-mu)/sd);}
ld tCritical95(int df){ld z=1.959963984540054L,v=max(1,df);return z+(z*z*z+z)/(4*v)+(5*powl(z,5)+16*powl(z,3)+3*z)/(96*v*v)+(3*powl(z,7)+19*powl(z,5)+17*powl(z,3)-15*z)/(384*powl(v,3));}

Stats computeStats(const vector<ld>& y){
    Stats s;s.n=y.size();if(y.empty())return s;
    s.mean=meanOf(y);s.median=quantile(y,.5);s.variance=varianceOf(y);s.sd=sqrtl(max<ld>(0,s.variance));
    auto[mi,ma]=minmax_element(y.begin(),y.end());s.minv=*mi;s.maxv=*ma;s.q1=quantile(y,.25);s.q3=quantile(y,.75);s.iqr=s.q3-s.q1;
    vector<ld>dev(y.size());for(size_t i=0;i<y.size();++i)dev[i]=fabsl(y[i]-s.median);s.mad=quantile(dev,.5);
    if(s.sd>1e-18L){ld m3=0,m4=0;for(ld v:y){ld z=(v-s.mean)/s.sd;m3+=z*z*z;m4+=z*z*z*z;}s.skew=m3/y.size();s.excessKurtosis=m4/y.size()-3;}
    if(y.size()>=3){ld num=0,den=0,dw=0;for(size_t i=1;i<y.size();++i){num+=(y[i]-s.mean)*(y[i-1]-s.mean);dw+=(y[i]-y[i-1])*(y[i]-y[i-1]);}for(ld v:y)den+=(v-s.mean)*(v-s.mean);if(den>0)s.lag1=num/den;s.durbinWatson=den>0?dw/den:numeric_limits<ld>::quiet_NaN();}
    if(y.size()>=8){int runs=0,last=0;for(ld v:y){int cur=v>s.median?1:(v<s.median?-1:0);if(cur==0)continue;if(cur!=last){++runs;last=cur;}}ld n1=count_if(y.begin(),y.end(),[&](ld v){return v>s.median;}),n2=count_if(y.begin(),y.end(),[&](ld v){return v<s.median;});if(n1>0&&n2>0){ld mu=1+2*n1*n2/(n1+n2),var=2*n1*n2*(2*n1*n2-n1-n2)/(powl((ld)n1+n2,2)*((ld)n1+n2-1));s.runsZ=(runs-mu)/sqrtl(max(var,1e-18L));}}
    if(y.size()>=3){s.jarqueBera=y.size()/6.0L*(s.skew*s.skew+s.excessKurtosis*s.excessKurtosis/4.0L);s.jarqueBeraP=exp(-s.jarqueBera/2);}
    if(s.n>=2&&s.sd>0){s.standardError=s.sd/sqrtl((ld)s.n);ld t=tCritical95(s.n-1);s.ci95Low=s.mean-t*s.standardError;s.ci95High=s.mean+t*s.standardError;}
    return s;
}

void printStats(const vector<ld>& y,const string&label="Dataset"){
    if(y.empty()) return;
    Stats s=computeStats(y);
    cout<<"\n================ STATISTICAL DIAGNOSTICS ================\n";
    cout<<"Variable           : "<<label<<"\n";
    cout<<"N                  : "<<s.n<<"\n";
    cout<<"Mean               : "<<fmt(s.mean)<<"\n";
    cout<<"Median             : "<<fmt(s.median)<<"\n";
    cout<<"Std. deviation     : "<<fmt(s.sd)<<"\n";
    cout<<"Min / Max          : "<<fmt(s.minv)<<" / "<<fmt(s.maxv)<<"\n";
    cout<<"Q1 / Q3 / IQR      : "<<fmt(s.q1)<<" / "<<fmt(s.q3)<<" / "<<fmt(s.iqr)<<"\n";
    cout<<"MAD                : "<<fmt(s.mad)<<"\n";
    cout<<"Skewness            : "<<fmt(s.skew)<<"\n";
    cout<<"Excess kurtosis     : "<<fmt(s.excessKurtosis)<<"\n";
    if(isfinite(s.lag1))cout<<"Lag-1 autocorr      : "<<fmt(s.lag1)<<"\n";
    if(isfinite(s.durbinWatson))cout<<"Durbin-Watson       : "<<fmt(s.durbinWatson)<<"\n";
    if(isfinite(s.runsZ))cout<<"Runs-test z         : "<<fmt(s.runsZ)<<"\n";
    if(isfinite(s.jarqueBera))cout<<"Jarque-Bera         : "<<fmt(s.jarqueBera)<<"  approx p="<<fmt(s.jarqueBeraP)<<"\n";
    if(isfinite(s.standardError))cout<<"Mean 95% CI         : ["<<fmt(s.ci95Low)<<", "<<fmt(s.ci95High)<<"]\n";

    cout<<"\nNormal model        : mu="<<fmt(s.mean)<<", sigma="<<fmt(s.sd)<<"\n";
    cout<<"Normal 95% band     : ["<<fmt(s.mean-1.96L*s.sd)<<", "<<fmt(s.mean+1.96L*s.sd)<<"]\n";

    bool binary=all_of(y.begin(),y.end(),[](ld v){return approxEqual(v,0,1e-12L,1e-12L)||approxEqual(v,1,1e-12L,1e-12L);});
    if(binary){
        ld p=s.mean;int n=s.n;int k=(int)llround(p*n);
        auto binomP=[&](int kk){if(kk<0||kk>n)return 0.0L;return expl(lgammal(n+1)-lgammal(kk+1)-lgammal(n-kk+1)+kk*logl(max(p,1e-18L))+(n-kk)*logl(max(1-p,1e-18L)));};
        cout<<"\nBinomial/Bernoulli  : p_hat="<<fmt(p)<<", trials observed="<<n<<"\n";
        cout<<"P(X="<<k<<")           : "<<fmt(binomP(k),6)<<"\n";
    }else{
        bool counts=all_of(y.begin(),y.end(),[](ld v){return v>=0&&isInteger(v);});
        if(counts){
            ld lambda=s.mean;long long k=(long long)llround(lambda);
            ld pk=(lambda==0?(k==0?1:0):expl(k*logl(lambda)-lambda-lgammal(k+1)));
            cout<<"\nPoisson diagnostic  : lambda_hat="<<fmt(lambda)<<"\n";
            cout<<"P(X="<<k<<")           : "<<fmt(pk,6)<<"\n";
            cout<<"Approx 95% count band: ["<<fmt(max<ld>(0,lambda-1.96L*sqrtl(max<ld>(lambda,1e-12L))))<<", "<<fmt(lambda+1.96L*sqrtl(max<ld>(lambda,1e-12L)))<<"]\n";
        }
    }
    if(s.n>=10){
        int bins=max(3,min(10,(int)floor(sqrt((ld)s.n))));vector<int>cnt(bins,0);ld range=s.maxv-s.minv;if(range<1e-18L)range=1;
        for(ld v:y){int b=(int)floor((v-s.minv)/range*bins);b=max(0,min(bins-1,b));cnt[b]++;}
        ld expected=(ld)s.n/bins,chi=0;for(int c:cnt)chi+=(c-expected)*(c-expected)/expected;
        ld p=chiSquarePApprox(chi,bins-1);
        cout<<"\nChi-square uniform-bin test: chi2="<<fmt(chi,5)<<", df="<<bins-1<<", approx p="<<fmt(p,6)<<"\n";
        cout<<"Note: this tests uniformity over observed bins, not randomness in general.\n";
    }
    cout<<"Adaptive deviation tolerance: "<<fmtFixed(adaptiveTolerancePct(y),3)<<"%\n";
    cout<<"=========================================================\n";
}

void printCompactFallbackDiagnostics(const vector<ld>& y,const string&why,ld usedTolerancePct=-1.0L){
    Stats s=computeStats(y);
    cout<<"\n[STATISTICAL FALLBACK DIAGNOSTICS]\n";
    cout<<"Reason             : "<<why<<"\n";
    cout<<"Mean               : "<<fmt(s.mean)<<"\n";
    cout<<"Std deviation      : "<<fmt(s.sd)<<"\n";
    cout<<"Median / MAD       : "<<fmt(s.median)<<" / "<<fmt(s.mad)<<"\n";
    cout<<"Validation tolerance: "<<fmtFixed(usedTolerancePct>=0?usedTolerancePct:adaptiveTolerancePct(y),3)<<"%\n";
    if(isfinite(s.lag1))cout<<"Lag-1 autocorr      : "<<fmt(s.lag1)<<"\n";
    if(isfinite(s.runsZ))cout<<"Runs-test z         : "<<fmt(s.runsZ)<<"\n";
    cout<<"=========================================================\n";
}

// ----------------------------- Randomness / generation ----------
// UI helpers are declared early because this section also contains the PRNG menu.
bool g_inputClosed = false;
bool g_lastInputCancelled = false;
void flushConsole() { cout << std::flush; }

string readLinePrompt(const string& prompt);
int askIntRange(const string& prompt, int lo, int hi, int def = -1);
bool askYesNo(const string& prompt, bool def = false);
vector<ld> parseNumberLine(string line);
vector<ld> readSequenceInteractive();
uint64_t splitmix64Next(uint64_t&state){uint64_t z=(state+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

void printRandomDiagnostics(const vector<ld>& y){
    cout<<"\n================ RANDOM / PRNG DIAGNOSTICS ===============\n";
    if(y.size()<6){cout<<"Need at least 6 observations for useful PRNG checks.\n";return;}
    auto l=detectLCG(vector<ld>(y.size()),y);
    auto x=detectXorShift32(vector<ld>(y.size()),y);
    if(l.valid)cout<<"Exact generator candidate: "<<l.name<<"\n"<<"Parameters/formula: "<<l.formula<<"\n";
    else if(x.valid)cout<<"Exact generator candidate: "<<x.name<<"\n"<<"Transition: "<<x.formula<<"\n";
    else cout<<"Exact common generator match: none among implemented LCG/xorshift32 checks.\n";
    Stats s=computeStats(y);
    cout<<"Mean / SD           : "<<fmt(s.mean)<<" / "<<fmt(s.sd)<<"\n";
    if(isfinite(s.lag1))cout<<"Lag-1 autocorr      : "<<fmt(s.lag1)<<"\n";
    if(isfinite(s.runsZ))cout<<"Runs-test z         : "<<fmt(s.runsZ)<<"\n";
    cout<<"Important           : statistical randomness tests cannot prove that a finite sequence is truly random.\n";
    cout<<"===========================================================\n";
}

void generateRandomSeries(){
    cout << "\n================ RANDOM / PRNG GENERATOR ================\n"
         << "1) std::mt19937 integer\n"
         << "2) Linear congruential generator\n"
         << "3) xorshift32\n"
         << "4) splitmix64\n"
         << "0) Back\n" << std::flush;
    const int generator = askIntRange("Choose generator (0-4): ", 0, 4, -1);
    if (g_inputClosed || generator == 0) return;

    const uint64_t seed = [&]() {
        while (true) {
            const string line = readLinePrompt("Seed (unsigned integer): ");
            if (g_inputClosed) return uint64_t{0};
            try {
                size_t pos = 0;
                const uint64_t value = stoull(line, &pos);
                if (pos == line.size()) return value;
            } catch (...) {
            }
            cout << "  Please enter a valid unsigned integer.\n";
        }
    }();
    if (g_inputClosed) return;

    const int count = askIntRange("How many values? (1-1000000): ", 1, 1000000, -1);
    if (g_inputClosed) return;

    cout << "Generated values:\n";
    if (generator == 1) {
        mt19937 rng(static_cast<uint32_t>(seed));
        uniform_int_distribution<uint32_t> dist(0, numeric_limits<uint32_t>::max());
        for (int i = 0; i < count; ++i) cout << dist(rng) << (i + 1 == count ? '\n' : ' ');
    } else if (generator == 2) {
        const auto parts = parseNumberLine(readLinePrompt("LCG parameters (m a c), entered as three unsigned integers: "));
        if (parts.size() != 3 || any_of(parts.begin(), parts.end(), [](ld v) { return !isfinite(v) || v < 0 || floorl(v) != v; })) {
            cout << "Invalid LCG parameters.\n";
            return;
        }
        const uint64_t modulus = static_cast<uint64_t>(parts[0]);
        const uint64_t multiplier = static_cast<uint64_t>(parts[1]);
        const uint64_t increment = static_cast<uint64_t>(parts[2]);
        if (modulus < 2 || multiplier >= modulus || increment >= modulus) {
            cout << "Invalid LCG parameters: require m>=2, 0<=a<m, 0<=c<m.\n";
            return;
        }
        uint64_t current = seed % modulus;
        for (int i = 0; i < count; ++i) {
            if (i) current = (multiplier * current + increment) % modulus;
            cout << current << (i + 1 == count ? '\n' : ' ');
        }
    } else if (generator == 3) {
        auto current = static_cast<uint32_t>(seed);
        for (int i = 0; i < count; ++i) {
            if (i) current = xorshift32(current);
            cout << current << (i + 1 == count ? '\n' : ' ');
        }
    } else if (generator == 4) {
        auto state = seed;
        for (int i = 0; i < count; ++i) cout << splitmix64Next(state) << (i + 1 == count ? '\n' : ' ');
    }
}

// Interactive UI forward declarations used by menus declared earlier.
bool parseIntStrict(const string& s, int& out);

// ----------------------------- Cipher / crypto lab ----------------
// Classical ciphers are included for education and interoperability. They are
// NOT modern password protection. For password-derived keys, PBKDF2-HMAC-SHA256
// is included; for authenticated modern encryption, an optional OpenSSL AES-256-GCM
// path can be enabled with ENGINE_USE_OPENSSL.
int hexValue(char c){if(c>='0'&&c<='9')return c-'0';c=(char)toupper((unsigned char)c);if(c>='A'&&c<='F')return c-'A'+10;return -1;}
long long parseBase(const string&s,int base){long long v=0;for(char c:s){if(isspace((unsigned char)c)||c=='_'||c=='-')continue;int d=(c>='0'&&c<='9')?c-'0':toupper((unsigned char)c)-'A'+10;if(d<0||d>=base)return -1;if(v>(numeric_limits<long long>::max()-d)/base)return numeric_limits<long long>::max();v=v*base+d;}return v;}
bool validInBase(const string&s,int base){bool any=false;for(char c:s){if(isspace((unsigned char)c)||c=='_'||c=='-')continue;int d=(c>='0'&&c<='9')?c-'0':toupper((unsigned char)c)-'A'+10;if(d<0||d>=base)return false;any=true;}return any;}
string hexEncode(const string&s){static const char*H="0123456789ABCDEF";string o;o.reserve(s.size()*2);for(unsigned char c:s){o+=H[c>>4];o+=H[c&15];}return o;}
string hexDecode(const string&s){string h;for(char c:s)if(!isspace((unsigned char)c))h+=c;if(h.size()%2)return{};string o;o.reserve(h.size()/2);for(size_t i=0;i<h.size();i+=2){int a=hexValue(h[i]),b=hexValue(h[i+1]);if(a<0||b<0)return{};o.push_back((char)((a<<4)|b));}return o;}
const string B64="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
string base64Encode(const string&s){string o;int val=0,valb=-6;for(unsigned char c:s){val=(val<<8)+c;valb+=8;while(valb>=0){o.push_back(B64[(val>>valb)&63]);valb-=6;}}if(valb>-6)o.push_back(B64[((val<<8)>>(valb+8))&63]);while(o.size()%4)o.push_back('=');return o;}
string base64Decode(const string&s){array<int,256>T{};T.fill(-1);for(int i=0;i<64;++i)T[(unsigned char)B64[i]]=i;string o;int val=0,valb=-8;for(char ch:s){if(isspace((unsigned char)ch))continue;if(ch=='=')break;if(T[(unsigned char)ch]<0)return{};val=(val<<6)+T[(unsigned char)ch];valb+=6;if(valb>=0){o.push_back((char)((val>>valb)&255));valb-=8;}}return o;}
string caesar(const string&s,int shift){string o=s;for(char&c:o)if(isalpha((unsigned char)c)){char a=isupper((unsigned char)c)?'A':'a';c=(char)(a+(c-a+shift%26+26)%26);}return o;}
string atbash(const string&s){string o=s;for(char&c:o)if(isalpha((unsigned char)c)){char a=isupper((unsigned char)c)?'A':'a';c=(char)(a+25-(c-a));}return o;}
int modInverse26(int a){a%=26;if(a<0)a+=26;for(int x=1;x<26;++x)if((a*x)%26==1)return x;return -1;}
string affineTransform(const string&s,int a,int b,bool decrypt){int inv=modInverse26(a);if(decrypt&&inv<0)return{};string o=s;for(char&c:o)if(isalpha((unsigned char)c)){char base=isupper((unsigned char)c)?'A':'a';int x=c-base;int y=decrypt?(inv*(x-b)%26+26)%26:(a*x+b)%26;c=(char)(base+y);}return o;}
string vigenereDecrypt(const string&s,const string&key){if(key.empty())return{};string o;size_t j=0;for(char c:s){if(isalpha((unsigned char)c)){int k=toupper((unsigned char)key[j%key.size()])-'A';if(k<0||k>25)return{};char a=isupper((unsigned char)c)?'A':'a';o.push_back((char)(a+(c-a-k+26)%26));++j;}else o.push_back(c);}return o;}
string xorWithKey(const string&s,const string&key){if(key.empty())return{};string o=s;for(size_t i=0;i<s.size();++i)o[i]=(char)((unsigned char)s[i]^(unsigned char)key[i%key.size()]);return o;}
string railFenceEncrypt(const string&s,int rails){if(rails<2||rails>=(int)s.size())return s;string o;int r=0,dir=1;vector<string>rows(rails);for(char c:s){rows[r].push_back(c);if(r==0)dir=1;else if(r==rails-1)dir=-1;r+=dir;}for(auto&v:rows)o+=v;return o;}
string railFenceDecrypt(const string&s,int rails){if(rails<2||rails>=(int)s.size())return s;vector<int>mark(s.size());int r=0,dir=1;for(size_t i=0;i<s.size();++i){mark[i]=r;if(r==0)dir=1;else if(r==rails-1)dir=-1;r+=dir;}vector<int>count(rails);for(int q:mark)++count[q];vector<string>rows(rails);size_t pos=0;for(int rr=0;rr<rails;++rr){rows[rr]=s.substr(pos,count[rr]);pos+=count[rr];}vector<int>idx(rails);string o;o.reserve(s.size());for(size_t i=0;i<s.size();++i){int rr=mark[i];o.push_back(rows[rr][idx[rr]++]);}return o;}
const unordered_map<char,string>& morseMap(){static const unordered_map<char,string>m={{'A',".-"},{'B',"-..."},{'C',"-.-."},{'D',"-.."},{'E',"."},{'F',"..-."},{'G',"--."},{'H',"...."},{'I',".."},{'J',".---"},{'K',"-.-"},{'L',".-.."},{'M',"--"},{'N',"-."},{'O',"---"},{'P',".--."},{'Q',"--.-"},{'R',".-."},{'S',"..."},{'T',"-"},{'U',"..-"},{'V',"...-"},{'W',".--"},{'X',"-..-"},{'Y',"-.--"},{'Z',"--.."},{'0',"-----"},{'1',".----"},{'2',"..---"},{'3',"...--"},{'4',"....-"},{'5',"....."},{'6',"-...."},{'7',"--..."},{'8',"---.."},{'9',"----."}};return m;}
string morseEncode(const string&s){auto&m=morseMap();string o;bool needSpace=false;for(char c:s){if(isspace((unsigned char)c)){if(!o.empty()&&o.back()!='/'&&o.back()!=' ')o.push_back('/');needSpace=false;continue;}auto it=m.find((char)toupper((unsigned char)c));if(it==m.end())continue;if(needSpace)o.push_back(' ');o+=it->second;needSpace=true;}return o;}
string morseDecode(const string&s){unordered_map<string,char>r;for(auto&kv:morseMap())r[kv.second]=kv.first;string o,token;for(size_t i=0;i<=s.size();++i){char c=i<s.size()?s[i]:' ';if(c=='/'){if(!token.empty()){auto it=r.find(token);if(it!=r.end())o.push_back(it->second);token.clear();}o.push_back(' ');}else if(isspace((unsigned char)c)){if(!token.empty()){auto it=r.find(token);if(it!=r.end())o.push_back(it->second);token.clear();}}else token.push_back(c);}return o;}

// ---------------- SHA-256 / HMAC / PBKDF2 ----------------------
struct SHA256Ctx{array<uint32_t,8>h{};uint64_t bitlen=0;array<uint8_t,64>buf{};size_t len=0;};
uint32_t rotr32(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
void sha256Init(SHA256Ctx&c){c.h={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};c.bitlen=0;c.len=0;}
void sha256Transform(SHA256Ctx&c,const uint8_t*d){static const uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};uint32_t w[64];for(int i=0;i<16;++i)w[i]=(uint32_t(d[4*i])<<24)|(uint32_t(d[4*i+1])<<16)|(uint32_t(d[4*i+2])<<8)|uint32_t(d[4*i+3]);for(int i=16;i<64;++i){uint32_t s0=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);uint32_t s1=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+s0+w[i-7]+s1;}uint32_t a=c.h[0],b=c.h[1],cc=c.h[2],d0=c.h[3],e=c.h[4],f=c.h[5],g=c.h[6],hh=c.h[7];for(int i=0;i<64;++i){uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25);uint32_t ch=(e&f)^((~e)&g);uint32_t t1=hh+S1+ch+K[i]+w[i];uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22);uint32_t maj=(a&b)^(a&cc)^(b&cc);uint32_t t2=S0+maj;hh=g;g=f;f=e;e=d0+t1;d0=cc;cc=b;b=a;a=t1+t2;}c.h[0]+=a;c.h[1]+=b;c.h[2]+=cc;c.h[3]+=d0;c.h[4]+=e;c.h[5]+=f;c.h[6]+=g;c.h[7]+=hh;}
void sha256Update(SHA256Ctx&c,const uint8_t*d,size_t n){for(size_t i=0;i<n;++i){c.buf[c.len++]=d[i];if(c.len==64){sha256Transform(c,c.buf.data());c.bitlen+=512;c.len=0;}}}
array<uint8_t,32> sha256Final(SHA256Ctx&c){size_t i=c.len;c.buf[i++]=0x80;if(i>56){while(i<64)c.buf[i++]=0;sha256Transform(c,c.buf.data());i=0;}while(i<56)c.buf[i++]=0;c.bitlen+=c.len*8;for(int j=7;j>=0;--j)c.buf[56+(7-j)]=(uint8_t)((c.bitlen>>(8*j))&0xFF);sha256Transform(c,c.buf.data());array<uint8_t,32>o{};for(int j=0;j<8;++j){o[4*j]=(uint8_t)(c.h[j]>>24);o[4*j+1]=(uint8_t)(c.h[j]>>16);o[4*j+2]=(uint8_t)(c.h[j]>>8);o[4*j+3]=(uint8_t)c.h[j];}return o;}
array<uint8_t,32> sha256Bytes(const string&s){SHA256Ctx c;sha256Init(c);sha256Update(c,(const uint8_t*)s.data(),s.size());return sha256Final(c);}
string bytesToHex(const vector<uint8_t>&b){static const char*H="0123456789abcdef";string o;o.reserve(b.size()*2);for(uint8_t q:b){o+=H[q>>4];o+=H[q&15];}return o;}
string sha256Hex(const string&s){auto a=sha256Bytes(s);return bytesToHex(vector<uint8_t>(a.begin(),a.end()));}
vector<uint8_t> hmacSha256Bytes(const vector<uint8_t>&key,const vector<uint8_t>&msg){vector<uint8_t>k=key;if(k.size()>64){auto h=sha256Bytes(string((const char*)k.data(),k.size()));k.assign(h.begin(),h.end());}k.resize(64,0);vector<uint8_t>ipad(64),opad(64);for(int i=0;i<64;++i){ipad[i]=k[i]^0x36;opad[i]=k[i]^0x5c;}SHA256Ctx c;sha256Init(c);sha256Update(c,ipad.data(),ipad.size());sha256Update(c,msg.data(),msg.size());auto ih=sha256Final(c);sha256Init(c);sha256Update(c,opad.data(),opad.size());sha256Update(c,ih.data(),ih.size());auto oh=sha256Final(c);return vector<uint8_t>(oh.begin(),oh.end());}
string hmacSha256Hex(const string&key,const string&msg){vector<uint8_t>k(key.begin(),key.end()),m(msg.begin(),msg.end());return bytesToHex(hmacSha256Bytes(k,m));}
string hmacSha256HexFromHexKey(const string&hexKey,const string&msg){string raw=hexDecode(hexKey);if(raw.empty()&&hexKey!="")return{};vector<uint8_t>k(raw.begin(),raw.end()),m(msg.begin(),msg.end());return bytesToHex(hmacSha256Bytes(k,m));}
vector<uint8_t> u32be(uint32_t v){return {(uint8_t)(v>>24),(uint8_t)(v>>16),(uint8_t)(v>>8),(uint8_t)v};}
vector<uint8_t> pbkdf2HmacSha256(const string&password,const vector<uint8_t>&salt,uint32_t iterations,size_t dkLen){vector<uint8_t>key(password.begin(),password.end());vector<uint8_t>dk;dk.reserve(dkLen);uint32_t blocks=(uint32_t)((dkLen+31)/32);for(uint32_t b=1;b<=blocks;++b){vector<uint8_t>sb=salt,bi=u32be(b);sb.insert(sb.end(),bi.begin(),bi.end());vector<uint8_t>u=hmacSha256Bytes(key,sb),t=u;for(uint32_t c=1;c<iterations;++c){u=hmacSha256Bytes(key,u);for(size_t j=0;j<t.size();++j)t[j]^=u[j];}dk.insert(dk.end(),t.begin(),t.end());}dk.resize(dkLen);return dk;}
vector<uint8_t> secureRandomBytes(size_t n){
    vector<uint8_t> o(n);
    if(n == 0) return o;
#ifdef _WIN32
    // Load Windows CNG dynamically so MinGW/CLion does not need a
    // hard link against bcrypt.lib. BCryptGenRandom remains the preferred
    // Windows system CSPRNG when bcrypt.dll is available.
    HMODULE h = LoadLibraryW(L"bcrypt.dll");
    if(h){
        using BCryptGenRandomFn = NTSTATUS (WINAPI *)(BCRYPT_ALG_HANDLE, PUCHAR, ULONG, ULONG);
        auto fn = reinterpret_cast<BCryptGenRandomFn>(GetProcAddress(h, "BCryptGenRandom"));
        if(fn && n <= static_cast<size_t>(numeric_limits<ULONG>::max())){
            const NTSTATUS status = fn(nullptr, o.data(), static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            FreeLibrary(h);
            if(status == 0) return o;
        }else{
            FreeLibrary(h);
        }
    }
#else
    ifstream f("/dev/urandom", ios::binary);
    if(f){
        f.read(reinterpret_cast<char*>(o.data()), static_cast<streamsize>(n));
        if(f.gcount() == static_cast<streamsize>(n)) return o;
    }
#endif
    // Final fallback for non-cryptographic convenience features.
    // Production secrets should use a vetted OS/library CSPRNG.
    random_device rd;
    for(size_t i = 0; i < n; ++i) o[i] = static_cast<uint8_t>(rd() & 0xFFu);
    return o;
}
vector<uint8_t> parseHexBytes(const string&s){string h;for(char c:s)if(!isspace((unsigned char)c))h+=c;if(h.size()%2)return{};vector<uint8_t>o;for(size_t i=0;i<h.size();i+=2){int a=hexValue(h[i]),b=hexValue(h[i+1]);if(a<0||b<0)return{};o.push_back((uint8_t)((a<<4)|b));}return o;}

ld englishChiSquare(const string&s,int shift){static const double f[26]={8.167,1.492,2.782,4.253,12.702,2.228,2.015,6.094,6.966,0.153,0.772,4.025,2.406,6.749,7.507,1.929,0.095,5.987,6.327,9.056,2.758,0.978,2.360,0.150,1.974,0.074};double c[26]={};double n=0;for(char c0:s)if(isalpha((unsigned char)c0)){int a=toupper((unsigned char)c0)-'A';int z=(a+shift+26)%26;c[z]++;n++;}if(n==0)return numeric_limits<ld>::infinity();double chi=0;for(int i=0;i<26;++i){double e=n*f[i]/100.0;chi+=(c[i]-e)*(c[i]-e)/max(e,1e-12);}return (ld)chi;}
ld indexOfCoincidence(const string&s){long long cnt[26]={};long long n=0;for(char c:s)if(isalpha((unsigned char)c)){cnt[toupper((unsigned char)c)-'A']++;n++;}if(n<2)return numeric_limits<ld>::quiet_NaN();long long num=0;for(long long v:cnt)num+=v*(v-1);return (ld)num/(n*(n-1));}
void hashIdentify(const string&s){string h;for(char c:s)if(!isspace((unsigned char)c))h+=c;bool hex=all_of(h.begin(),h.end(),[](char c){return isxdigit((unsigned char)c);});cout<<"Hash/encoding shape hints:\n";if(hex){if(h.size()==32)cout<<"  32 hex chars -> MD5-shaped (not proof).\n";if(h.size()==40)cout<<"  40 hex chars -> SHA-1-shaped (not proof).\n";if(h.size()==64)cout<<"  64 hex chars -> SHA-256-shaped (not proof).\n";if(h.size()==128)cout<<"  128 hex chars -> SHA-512-shaped (not proof).\n";}if(s.size()%4==0&&s.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/= \t\r\n")==string::npos)cout<<"  Base64-shaped text.\n";}

#ifdef ENGINE_USE_OPENSSL
string aesGcmEncryptOpenSSL(const string&keyHex,const string&plaintext,const string&ivHex){vector<uint8_t>key=parseHexBytes(keyHex),iv=parseHexBytes(ivHex);if(key.size()!=32||iv.size()!=12)return{};vector<uint8_t>out(plaintext.size()+16);int l1=0,l2=0;EVP_CIPHER_CTX*ctx=EVP_CIPHER_CTX_new();if(!ctx)return{};string result;do{if(EVP_EncryptInit_ex(ctx,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)!=1)break;if(EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_IVLEN,(int)iv.size(),nullptr)!=1)break;if(EVP_EncryptInit_ex(ctx,nullptr,nullptr,key.data(),iv.data())!=1)break;if(EVP_EncryptUpdate(ctx,out.data(),&l1,(const uint8_t*)plaintext.data(),(int)plaintext.size())!=1)break;if(EVP_EncryptFinal_ex(ctx,out.data()+l1,&l2)!=1)break;vector<uint8_t>ct(out.begin(),out.begin()+l1+l2),tag(16);if(EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_GET_TAG,16,tag.data())!=1)break;result=hexEncode(string((const char*)ct.data(),ct.size()))+":"+hexEncode(string((const char*)tag.data(),tag.size()));}while(false);EVP_CIPHER_CTX_free(ctx);return result;}
string aesGcmDecryptOpenSSL(const string&keyHex,const string&cipherHex,const string&ivHex,const string&tagHex){vector<uint8_t>key=parseHexBytes(keyHex),iv=parseHexBytes(ivHex),ct=parseHexBytes(cipherHex),tag=parseHexBytes(tagHex);if(key.size()!=32||iv.size()!=12||tag.size()!=16)return{};vector<uint8_t>out(ct.size()+16);int l1=0,l2=0;EVP_CIPHER_CTX*ctx=EVP_CIPHER_CTX_new();if(!ctx)return{};string result;do{if(EVP_DecryptInit_ex(ctx,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)!=1)break;if(EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_IVLEN,(int)iv.size(),nullptr)!=1)break;if(EVP_DecryptInit_ex(ctx,nullptr,nullptr,key.data(),iv.data())!=1)break;if(EVP_DecryptUpdate(ctx,out.data(),&l1,ct.data(),(int)ct.size())!=1)break;if(EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_TAG,16,tag.data())!=1)break;if(EVP_DecryptFinal_ex(ctx,out.data()+l1,&l2)!=1)break;result=string((const char*)out.data(),l1+l2);}while(false);EVP_CIPHER_CTX_free(ctx);return result;}
#endif

void analyzeCipher(const string&input){
    cout<<"\n================ CRYPTO / ENCODING ANALYSIS ==============\n";
    string compact;for(char c:input)if(!isspace((unsigned char)c))compact.push_back(c);
    for(auto[base,name]:vector<pair<int,string>>{{2,"binary"},{8,"octal"},{10,"decimal"},{16,"hexadecimal"}})if(validInBase(compact,base))cout<<"As "<<name<<": decimal = "<<parseBase(compact,base)<<"\n";
    string hx=hexDecode(input);if(!hx.empty())cout<<"Hex -> text       : "<<hx<<"\n";
    string b64=base64Decode(input);if(!b64.empty()&&base64Encode(b64)==compact)cout<<"Base64 -> text     : "<<b64<<"\n";
    cout<<"Morse-shaped input  : "<<(input.find('.')!=string::npos||input.find('-')!=string::npos?morseDecode(input):"(not inferred)")<<"\n";
    cout<<"Atbash               : "<<atbash(input)<<"\n";
    cout<<"Caesar best shifts   :\n";vector<pair<ld,int>>scores;for(int sh=0;sh<26;++sh)scores.push_back({englishChiSquare(input,sh),sh});sort(scores.begin(),scores.end());for(int i=0;i<min(5,(int)scores.size());++i)cout<<"  shift "<<scores[i].second<<" chi2="<<fmt(scores[i].first,3)<<" -> "<<caesar(input,scores[i].second)<<"\n";
    cout<<"IC (English-text diagnostic): "<<fmt(indexOfCoincidence(input),6)<<"\n";
    hashIdentify(input);
    cout<<"===========================================================\n";
}

void passwordKdfTool(){
    cout<<"\n================ PASSWORD KEY DERIVATION ==================\n";
    cout<<"A password is NOT itself a strong encryption key. This tool derives keying material using PBKDF2-HMAC-SHA256.\n";
    string pw=readLinePrompt("Password/passphrase: ");if(g_inputClosed)return;
    string saltText=readLinePrompt("Salt text (blank = generate 16-byte demo salt): ");if(g_inputClosed)return;
    vector<uint8_t>salt;if(saltText.empty())salt=secureRandomBytes(16);else salt.assign(saltText.begin(),saltText.end());
    int iterations=askIntRange("PBKDF2 iterations [600000]: ",1,5000000,600000);
    int bytes=askIntRange("Derived key bytes [32]: ",1,64,32);
    auto dk=pbkdf2HmacSha256(pw,salt,(uint32_t)iterations,(size_t)bytes);
    cout<<"Salt (hex)          : "<<bytesToHex(salt)<<"\n";
    cout<<"Iterations          : "<<iterations<<"\n";
    cout<<"Derived key (hex)   : "<<bytesToHex(dk)<<"\n";
    cout<<"Record example      : pbkdf2-sha256$"<<iterations<<"$"<<bytesToHex(salt)<<"$"<<bytesToHex(dk)<<"\n";
    cout<<"WARNING             : Store the salt + work factor with the verifier. Do not use a fast hash alone for password storage.\n";
}


void printCryptoTheory(){
    cout<<"\n================ CRYPTOGRAPHY THEORY ======================\n";
    cout<<"Caesar / ROT      : monoalphabetic shift; useful for learning substitution, trivially breakable by brute force.\n";
    cout<<"Atbash            : fixed mirror substitution A<->Z, B<->Y, etc.; no secret key.\n";
    cout<<"Vigenere          : repeated-key polyalphabetic substitution; key-length leakage can be studied with frequency/IC analysis.\n";
    cout<<"Affine            : maps letters with y=(a*x+b) mod 26; a must be invertible modulo 26.\n";
    cout<<"XOR               : bitwise combination. A repeated short XOR key is not secure; a true one-time pad requires a secret random key as long as the message and never reuses it.\n";
    cout<<"Hash (SHA-256)    : one-way digest for integrity/fingerprints; it is not encryption and should not be used alone for password storage.\n";
    cout<<"HMAC-SHA256       : keyed authentication tag built from a hash; provides message authentication/integrity when the key is secret.\n";
    cout<<"PBKDF2-HMAC-SHA256: deliberately repeats HMAC many times to turn a password/passphrase into keying material; a unique salt makes precomputation/rainbow-table reuse harder.\n";
    cout<<"AES-256-GCM       : modern authenticated symmetric encryption; AES supplies confidentiality, GCM supplies authentication/tagging. Requires a unique nonce/IV per key.\n";
    cout<<"Security boundary  : classical ciphers are educational. For production cryptography, use a vetted library such as OpenSSL/libsodium rather than hand-rolling primitives.\n";
    cout<<"===========================================================\n";
}

void cipherMenu(){
    cout<<"\n================ CRYPTO / ENCODING LAB ====================\n"
         <<"1) Auto analysis\n"
         <<"2) Caesar / ROT + English frequency ranking\n"
         <<"3) Vigenere decrypt + key-length diagnostics\n"
         <<"4) XOR encrypt/decrypt (educational, NOT secure alone)\n"
         <<"5) Base64 encode/decode\n"
         <<"6) Hex encode/decode\n"
         <<"7) Morse encode/decode\n"
         <<"8) Atbash\n"
         <<"9) Affine cipher\n"
         <<"10) Rail Fence cipher\n"
         <<"11) SHA-256\n"
         <<"12) HMAC-SHA256\n"
         <<"13) Password -> PBKDF2-HMAC-SHA256 key\n"
         <<"14) Generate random key bytes (demo utility)\n"
         <<"15) Cryptography theory / how the tools work\n"
#ifdef ENGINE_USE_OPENSSL
         <<"16) AES-256-GCM (OpenSSL build)\n"
#endif
         <<"0) Back\n"<<std::flush;
    int op=askIntRange("> ",0,
#ifdef ENGINE_USE_OPENSSL
                       16,
#else
                       15,
#endif
                       0);if(g_inputClosed||op==0)return;
    if(op==1){string in=readLinePrompt("Input: ");analyzeCipher(in);}
    else if(op==2){string in=readLinePrompt("Text/ciphertext: ");vector<pair<ld,int>>sc;for(int s=0;s<26;++s)sc.push_back({englishChiSquare(in,s),s});sort(sc.begin(),sc.end());for(auto [q,sh]:sc)cout<<"shift "<<setw(2)<<sh<<" chi2="<<fmt(q,3)<<" : "<<caesar(in,sh)<<"\n";}
    else if(op==3){string in=readLinePrompt("Ciphertext: ");string key=readLinePrompt("Key: ");cout<<"Plaintext: "<<vigenereDecrypt(in,key)<<"\n";cout<<"Index of coincidence: "<<fmt(indexOfCoincidence(in),6)<<"\n";}
    else if(op==4){string in=readLinePrompt("Text: ");string key=readLinePrompt("Key: ");string out=xorWithKey(in,key);cout<<"XOR hex: "<<hexEncode(out)<<"\n";}
    else if(op==5){string in=readLinePrompt("Text (encode) or Base64 (decode): ");string d=base64Decode(in);string compact;for(char c:in)if(!isspace((unsigned char)c))compact+=c;if(!d.empty()&&base64Encode(d)==compact)cout<<"Decoded: "<<d<<"\n";else cout<<"Encoded: "<<base64Encode(in)<<"\n";}
    else if(op==6){string in=readLinePrompt("Text or hex: ");string d=hexDecode(in);if(!d.empty())cout<<"Decoded: "<<d<<"\n";else cout<<"Encoded: "<<hexEncode(in)<<"\n";}
    else if(op==7){string in=readLinePrompt("Morse or plain text (letters/digits for encode): ");if(in.find('.')!=string::npos||in.find('-')!=string::npos)cout<<"Decoded: "<<morseDecode(in)<<"\n";else cout<<"Encoded: "<<morseEncode(in)<<"\n";}
    else if(op==8){cout<<"Atbash: "<<atbash(readLinePrompt("Text: "))<<"\n";}
    else if(op==9){string in=readLinePrompt("Text: ");int a=askIntRange("a (coprime to 26): ",1,25,5),b=askIntRange("b: ",0,25,8);string out=affineTransform(in,a,b,false);cout<<"Ciphertext: "<<out<<"\nPlaintext: "<<affineTransform(out,a,b,true)<<"\n";}
    else if(op==10){string in=readLinePrompt("Text: ");int r=askIntRange("Rails (2-20): ",2,20,3);string c=railFenceEncrypt(in,r);cout<<"Encrypted: "<<c<<"\nDecrypted: "<<railFenceDecrypt(c,r)<<"\n";}
    else if(op==11){string in=readLinePrompt("Message: ");cout<<"SHA-256: "<<sha256Hex(in)<<"\n";}
    else if(op==12){string key=readLinePrompt("HMAC key: ");string msg=readLinePrompt("Message: ");cout<<"HMAC-SHA256: "<<hmacSha256Hex(key,msg)<<"\n";}
    else if(op==13){passwordKdfTool();}
    else if(op==14){int bytes=askIntRange("Key bytes (8-64): ",8,64,32);cout<<"Random key hex: "<<bytesToHex(secureRandomBytes((size_t)bytes))<<"\n";cout<<"Use a vetted OS/library CSPRNG for production secrets.\n";}
    else if(op==15){printCryptoTheory();}
#ifdef ENGINE_USE_OPENSSL
    else if(op==16){
        cout<<"AES-256-GCM expects a 64-hex-char key and 24-hex-char IV.\n";
        int mode=askIntRange("1) Encrypt  2) Decrypt\n> ",1,2,1);
        string key=readLinePrompt("Key hex: "); string iv=readLinePrompt("12-byte IV hex: ");
        if(mode==1){
            string pt=readLinePrompt("Plaintext: "); string out=aesGcmEncryptOpenSSL(key,pt,iv);
            if(out.empty()) cout<<"Encryption failed.\n"; else {auto pos=out.find(':');cout<<"Ciphertext hex: "<<out.substr(0,pos)<<"\nTag hex       : "<<out.substr(pos+1)<<"\n";}
        }else{
            string ct=readLinePrompt("Ciphertext hex: "); string tag=readLinePrompt("Tag hex: ");
            string pt=aesGcmDecryptOpenSSL(key,ct,iv,tag); if(pt.empty()&& !ct.empty()) cout<<"Decryption/authentication failed or plaintext is empty.\n"; else cout<<"Plaintext: "<<pt<<"\n";
        }
    }
#endif
}

// ----------------------------- Input helpers --------------------
vector<ld> parseNumberLine(string line){
    // Flexible numeric parser: accepts spaces, commas, semicolons, tabs,
    // parentheses/brackets, and simple labels such as x=1,y=393.
    if(line.size()>=3 && (unsigned char)line[0]==0xEF && (unsigned char)line[1]==0xBB && (unsigned char)line[2]==0xBF) line.erase(0,3);
    const auto comment=line.find('#'); if(comment!=string::npos) line.erase(comment);
    for(char& c:line){
        if(c==','||c==';'||c=='\t'||c=='('||c==')'||c=='['||c==']'||c=='{'||c=='}'||c==':'||c=='='||c=='"'||c=='\'') c=' ';
    }
    stringstream ss(line); vector<ld> values; string token;
    auto isLabel=[](const string&t){
        string q=t; transform(q.begin(),q.end(),q.begin(),[](unsigned char c){return (char)tolower(c);});
        static const set<string>labels={"x","y","z","t","time","value","values","val","data","obs","observation","index","n","row","col","a","b","c"};
        return labels.count(q)>0;
    };
    while(ss>>token){
        try{
            size_t pos=0;ld value=stold(token,&pos);
            if(pos==token.size()&&isfinite(value)){values.push_back(value);continue;}
            size_t q=0;while(q<token.size()&&isalpha((unsigned char)token[q]))++q;
            if(q>0&&q<token.size()&&q<=6){
                string lead=token.substr(0,q);
                if(isLabel(lead)){
                    size_t p2=0;ld v=stold(token.substr(q),&p2);
                    if(p2==token.size()-q&&isfinite(v)){values.push_back(v);continue;}
                }
            }
            if(isLabel(token))continue;
            return {};
        }catch(...){ if(isLabel(token))continue; return {}; }
    }
    return values;
}

vector<vector<ld>> readMultivariateRows(int k){
    vector<vector<ld>> rows;string line;
    cout<<"Enter rows; parentheses/commas/spaces are accepted. Blank line ends input.\n";
    while(getline(cin,line)){
        if(line.find_first_not_of(" \t\r\n")==string::npos)break;
        vector<ld> vals=parseNumberLine(line);
        if(vals.empty())continue;
        if((int)vals.size()%k!=0){cout<<"  Ignored malformed line: found "<<vals.size()<<" values; expected a multiple of "<<k<<".\n";continue;}
        for(size_t i=0;i<vals.size();i+=k) rows.emplace_back(vals.begin()+i,vals.begin()+i+k);
    }
    return rows;
}


int quietSystem(const string& cmd){
#ifdef _WIN32
    return system((cmd+" >NUL 2>&1").c_str());
#else
    return system((cmd+" >/dev/null 2>&1").c_str());
#endif
}
bool commandExists(const string& cmd){
#ifdef _WIN32
    return quietSystem("where "+cmd)==0;
#else
    return quietSystem("command -v "+cmd)==0;
#endif
}
string preferredPython(){
#ifdef _WIN32
    if(commandExists("python")) return "python";
    if(commandExists("py")) return "py -3";
#else
    if(commandExists("python3")) return "python3";
    if(commandExists("python")) return "python";
#endif
    return {};
}

string shellQuote(const string&s){
#ifdef _WIN32
    string out="\"";
    for(char c:s){ if(c=='\"') out += "\\\""; else out.push_back(c); }
    out += "\""; return out;
#else
    string out="'";
    for(char c:s){ if(c=='\'') out += "'\\''"; else out.push_back(c); }
    out += "'"; return out;
#endif
}

void writePlotFiles(const vector<Point>&pts,const Model*model,const string&title,bool is3D,int horizon,ld forecastStep=1.0L){
    ofstream dat("sequence_plot_data.csv");
    dat<<"x,y,z,fit,next,kind\n";
    for(size_t i=0;i<pts.size();++i){
        ld fit=numeric_limits<ld>::quiet_NaN();
        if(model&&model->predict)fit=model->predict(pts[i].x);
        ld nx=numeric_limits<ld>::quiet_NaN();
        if(i+1==pts.size()&&model&&model->predict)nx=model->predict(pts[i].x+forecastStep);
        dat<<setprecision(21)<<pts[i].x<<","<<pts[i].y<<","<<pts[i].z<<","<<fit<<","<<nx<<",actual\n";
    }
    if(!is3D && model&&model->predict&&horizon>0&&!pts.empty()) {
        for(int h=1;h<=horizon;++h){
            ld fx=pts.back().x+forecastStep*h,fy=model->predict(fx);
            if(!isfinite(fy)) continue;
            dat<<setprecision(21)<<fx<<","<<fy<<",0,NaN,"<<fy<<",forecast\n";
        }
    }

    if(is3D){
        ofstream py("sequence_plot.py");
        py<<"formula="<<pyQuote(title)<<"\n";
        py<<R"PY(import csv, math
try:
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d import Axes3D
except Exception: raise SystemExit(2)
rows=list(csv.DictReader(open('sequence_plot_data.csv')))
x=[float(r['x']) for r in rows]; y=[float(r['y']) for r in rows]; z=[float(r['z']) for r in rows]
fig=plt.figure(figsize=(12,8)); ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,z,s=28,label='Actual points'); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('3D data' + ('\n'+formula if formula else '')); ax.legend(); plt.tight_layout(); plt.savefig('sequence_analysis_3d.png',dpi=160)
)PY";
        py.close();
        int rc=-1; { string py=preferredPython(); if(!py.empty()) rc=quietSystem(py+" sequence_plot.py"); }
        if(rc!=0&&quietSystem("gnuplot --version")==0){
            ofstream gp("sequence_plot3d.gnu");
            gp<<"set terminal pngcairo size 1200,800\nset output 'sequence_analysis_3d.png'\nset title '3D data'\nsplot 'sequence_plot_data.csv' using 1:2:3 with points pt 7 title 'Actual'\n";gp.close();
            quietSystem("gnuplot sequence_plot3d.gnu");
        }
        return;
    }

    ofstream py("sequence_plot.py");
    py<<"formula="<<pyQuote(title)<<"\n";
    py<<R"PY(import csv, math
try:
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except Exception: raise SystemExit(2)
rows=list(csv.DictReader(open('sequence_plot_data.csv')))
actual=[r for r in rows if r.get('kind')=='actual']; forecast=[r for r in rows if r.get('kind')=='forecast']
x=[float(r['x']) for r in actual]; y=[float(r['y']) for r in actual]; fit=[]
for r in actual:
 try:
  v=float(r['fit']); fit.append(v if math.isfinite(v) else math.nan)
 except Exception: fit.append(math.nan)
fx=[float(r['x']) for r in forecast]; fy=[float(r['y']) for r in forecast]
fig,ax=plt.subplots(figsize=(12,7)); ax.scatter(x,y,s=34,label='Actual data')
if any(math.isfinite(v) for v in fit): ax.plot(x,fit,linewidth=2.2,label='Validated fitted model')
if fy:
 ax.plot([x[-1]]+fx,[y[-1]]+fy,marker='*',linewidth=2.2,linestyle='--',label='Forecast continuation')
 ax.scatter(fx,fy,s=70,marker='*')
ax.set_xlabel('x / sequence index'); ax.set_ylabel('y'); ax.grid(True,alpha=.25)
ax.set_title('Sequence / curve analysis' + ('\n'+formula if formula else '')); ax.legend(); fig.tight_layout(); fig.savefig('sequence_analysis.png',dpi=170)
)PY";
    py.close();
    int rc=-1; { string py=preferredPython(); if(!py.empty()) rc=quietSystem(py+" sequence_plot.py"); }
    if(rc!=0&&quietSystem("gnuplot --version")==0){
        ofstream gp("sequence_plot.gnu");
        gp<<"set terminal pngcairo size 1200,800\nset output 'sequence_analysis.png'\nset title 'Sequence / curve analysis'\nset grid\nplot 'sequence_plot_data.csv' using 1:2 with points pt 7 title 'Actual', '' using 1:4 with lines lw 2 title 'Fitted'\n";gp.close();
        quietSystem("gnuplot sequence_plot.gnu");
    }
}

void asciiPlot(const vector<Point>&pts,const Model*model){
    if(pts.empty()) return;
    const int W=84,H=24;ld xmin=pts.front().x,xmax=pts.back().x,ymin=pts.front().y,ymax=pts.front().y;for(auto&p:pts){ymin=min(ymin,p.y);ymax=max(ymax,p.y);}if(fabsl(xmax-xmin)<EPS)xmax=xmin+1;if(fabsl(ymax-ymin)<EPS)ymax=ymin+1;vector<string>g(H,string(W,' '));auto put=[&](ld x,ld y,char ch){int c=(int)llround((x-xmin)/(xmax-xmin)*static_cast<ld>(W-1));int r=(int)llround((ymax-y)/(ymax-ymin)*static_cast<ld>(H-1));if(c>=0&&c<W&&r>=0&&r<H)g[r][c]=(g[r][c]==' ')?ch:'*';};for(auto&p:pts)put(p.x,p.y,'o');if(model&&model->predict)for(int i=0;i<W;++i){ld x=xmin+(xmax-xmin)*static_cast<ld>(i)/static_cast<ld>(W-1),y=model->predict(x);if(isfinite(y))put(x,y,'~');}for(auto&row:g)cout<<"|"<<row<<"|\n";cout<<"Legend: o=actual, ~=fitted, *=overlap\n";
}

void writeMultivariatePlotFiles(const vector<vector<ld>>& rows, const vector<vector<ld>>& futureRows) {
    if(rows.empty() || rows[0].empty()) return;
    ofstream dat("multivariate_forecast.csv");
    dat<<"kind,index";
    for(size_t j=0;j<rows[0].size();++j)dat<<",var"<<j+1;
    dat<<"\n";
    for(size_t i=0;i<rows.size();++i){dat<<"actual,"<<i+1;for(ld v:rows[i])dat<<","<<setprecision(18)<<(double)v;dat<<"\n";}
    for(size_t i=0;i<futureRows.size();++i){dat<<"forecast,"<<rows.size()+i+1;for(ld v:futureRows[i])dat<<","<<setprecision(18)<<(double)v;dat<<"\n";}
    ofstream py("multivariate_plot.py");
    py<<R"PY(import csv, math
try:
 import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
except Exception: raise SystemExit(2)
rows=list(csv.reader(open('multivariate_forecast.csv')))
header=rows[0]; data=rows[1:]
vars=len(header)-2
fig,ax=plt.subplots(figsize=(12,7))
for j in range(vars):
    xa=[];ya=[];xf=[];yf=[]
    for r in data:
        if r[0]=='actual': xa.append(float(r[1])); ya.append(float(r[2+j]))
        else: xf.append(float(r[1])); yf.append(float(r[2+j]))
    ax.plot(xa,ya,marker='o',label=f'Variable {j+1} actual')
    if yf: ax.plot(xf,yf,marker='*',linestyle='--',label=f'Variable {j+1} forecast')
ax.set_xlabel('Row index'); ax.set_ylabel('Value'); ax.set_title('Multivariate chained forecast'); ax.grid(True,alpha=.25); ax.legend(ncol=2); fig.tight_layout(); fig.savefig('multivariate_analysis.png',dpi=170)
)PY";
    py.close();
    int rc=-1; { string py=preferredPython(); if(!py.empty()) rc=quietSystem(py+" multivariate_plot.py"); }
    if(rc!=0 && quietSystem("gnuplot --version")==0){
        ofstream gp("multivariate_plot.gnu");
        gp<<"set terminal pngcairo size 1200,800\nset output 'multivariate_analysis.png'\nset title 'Multivariate chained forecast'\nset grid\n";
        const auto cols = sizeToInt(rows[0].size()) - 2;
        gp<<"plot ";
        for(int j=0;j<cols;++j){if(j)gp<<", ";gp<<"'multivariate_forecast.csv' using 2:"<<j+3<<" every ::1 with linespoints title 'Variable "<<j+1<<"'";} gp<<"\n";
        gp.close();quietSystem("gnuplot multivariate_plot.gnu");
    }
}

// ----------------------------- Output helpers -------------------
void printValidationSummary(const Model&m){
    if(!m.metrics.nValid)return;
    cout<<"VALIDATION MODE     : "<<(m.metrics.rollingValidated?"Rolling / moving-window origin":"Validation")<<"\n";
    cout<<"VALIDATION WINDOWS  : "<<m.metrics.validationBlocks<<"\n";
    cout<<"VALIDATION RMSE     : "<<fmt(m.metrics.rmse)<<"\n";
    cout<<"VALIDATION MAE      : "<<fmt(m.metrics.mae)<<"\n";
    cout<<"VALIDATION R^2      : "<<fmt(m.metrics.r2,6)<<"\n";
    cout<<"NORMALIZED RMSE     : "<<fmt(m.metrics.normalizedRMSE,6)<<"\n";
    cout<<"VALIDATION HIT RATE : "<<fmtFixed(m.metrics.hitRate*100,3)<<"%\n";
    cout<<"DIRECTIONAL ACCURACY: "<<(isfinite(m.metrics.directionalAccuracy)?fmtFixed(m.metrics.directionalAccuracy*100,3)+"%":"NaN")<<"\n";
    cout<<"CORRELATION         : "<<(isfinite(m.metrics.correlation)?fmt(m.metrics.correlation,6):"NaN")<<"\n";
    cout<<"SMAPE               : "<<fmtFixed(m.metrics.smapePct,3)<<"%\n";
    cout<<"MEAN DEVIATION      : "<<fmtFixed(m.metrics.meanDeviationPct,3)<<"%\n";
    cout<<"P90 DEVIATION       : "<<fmtFixed(m.metrics.p90DeviationPct,3)<<"%\n";
    cout<<"MAX DEVIATION       : "<<fmtFixed(m.metrics.maxDeviationPct,3)<<"%\n";
    cout<<"WAPE                : "<<fmtFixed(m.metrics.wapePct,3)<<"%\n";
    cout<<"MASE                : "<<fmt(m.metrics.mase,6)<<"\n";
    cout<<"MEDIAN DEVIATION    : "<<fmtFixed(m.metrics.medianDeviationPct,3)<<"%\n";
    cout<<"BLOCK PASS RATE     : "<<fmtFixed(m.metrics.blockPassRate*100,3)<<"%\n";
    cout<<"SCORE COMPONENTS    : RMSE="<<fmtFixed(m.metrics.scoreRMSE*100,1)<<" MAE="<<fmtFixed(m.metrics.scoreMAE*100,1)<<" P90="<<fmtFixed(m.metrics.scoreP90*100,1)<<" R2="<<fmtFixed(m.metrics.scoreR2*100,1)<<" DIR="<<fmtFixed(m.metrics.scoreDirection*100,1)<<" CORR="<<fmtFixed(m.metrics.scoreCorrelation*100,1)<<" STAB="<<fmtFixed(m.metrics.scoreStability*100,1)<<"\n";
    if(!m.metrics.blockMeanDeviationPct.empty()){
        cout<<"ROLLING MEAN DEVS   : [";for(size_t i=0;i<m.metrics.blockMeanDeviationPct.size()&&i<12;++i){if(i)cout<<", ";cout<<fmt(m.metrics.blockMeanDeviationPct[i],3);}if(m.metrics.blockMeanDeviationPct.size()>12)cout<<", ...";cout<<"]\n";
        cout<<"ROLLING MAX DEVS    : [";for(size_t i=0;i<m.metrics.blockMaxDeviationPct.size()&&i<12;++i){if(i)cout<<", ";cout<<fmt(m.metrics.blockMaxDeviationPct[i],3);}if(m.metrics.blockMaxDeviationPct.size()>12)cout<<", ...";cout<<"]\n";
    }
}

void printAnalysis(const Analysis&a,int horizon){
    cout<<"\n================ CORE LOGIC ANALYSIS ================\n";
    cout<<"VALIDATION TOLERANCE: "<<fmtFixed(a.tolerancePct,3)<<"%\n";
    cout<<"EVIDENCE SCORE      : "<<fixed<<setprecision(2)<<(double)a.evidenceScore<<"%\n";
    cout<<"CATEGORY            : "<<a.category<<"\n";
    cout<<"CORE LOGIC          : "<<a.logic<<"\n";
    if(!a.formula.empty())cout<<"MATHEMATICAL / ML MODEL: "<<a.formula<<"\n";
    if(!a.model.approxRealRoots.empty()){
        cout<<"APPROX REAL ROOTS   : ";
        for(size_t i=0;i<a.model.approxRealRoots.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.model.approxRealRoots[i],6);}
        if(a.model.approxRealRoots.size()>20)cout<<", ...";
        cout<<"\n";
    }
    if(!a.model.approxDerivativeRoots.empty()){
        cout<<"APPROX DERIVATIVE ROOTS: ";
        for(size_t i=0;i<a.model.approxDerivativeRoots.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.model.approxDerivativeRoots[i],6);}
        if(a.model.approxDerivativeRoots.size()>20)cout<<", ...";
        cout<<"\n";
    }
    if(!a.model.approxInflectionRoots.empty()){
        cout<<"APPROX INFLECTION ROOTS: ";
        for(size_t i=0;i<a.model.approxInflectionRoots.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.model.approxInflectionRoots[i],6);}
        if(a.model.approxInflectionRoots.size()>20)cout<<", ...";
        cout<<"\n";
    }
    if(!a.observedZeroCrossings.empty()){cout<<"NUMERICAL DATA ROOTS   : ";for(size_t i=0;i<a.observedZeroCrossings.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.observedZeroCrossings[i],6);}cout<<"\n";}
    if(!a.observedTurningPoints.empty()){cout<<"OBSERVED EXTREMA       : ";for(size_t i=0;i<a.observedTurningPoints.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.observedTurningPoints[i],6);}cout<<"\n";}
    if(!a.observedInflectionPoints.empty()){cout<<"OBSERVED INFLECTIONS   : ";for(size_t i=0;i<a.observedInflectionPoints.size()&&i<20;++i){if(i)cout<<", ";cout<<fmt(a.observedInflectionPoints[i],6);}cout<<"\n";}
    if(!a.calculusSummary.empty())cout<<"CALCULUS SUMMARY       : "<<a.calculusSummary<<"\n";
    if(a.shapeBased)cout<<"FORECAST STATUS        : shape-guided heuristic (not deterministic)\n";
    if(a.found){
        cout<<"PREDICTED NEXT "<<horizon<<" : ";for(size_t i=0;i<a.predictions.size();++i){if(i)cout<<", ";cout<<fmt(a.predictions[i]);}cout<<"\n";
        if(!a.predictions.empty())cout<<"NEXT TERM           : "<<fmt(a.predictions.front())<<"\n";
        if(!a.predictionLow.empty())cout<<"EMPIRICAL 90% ERROR BAND: first="<<fmt(a.predictionLow.front())<<" .. "<<fmt(a.predictionHigh.front())<<" (validation-residual band, not a formal CI)\n";
        if(!a.changePoints.empty()){cout<<"CHANGE-POINT CANDIDATES: ";for(size_t i=0;i<a.changePoints.size()&&i<10;++i){if(i)cout<<", ";cout<<a.changePoints[i];}cout<<"\n";}
        if(a.model.exact){cout<<"FIT DEVIATION       : exact / 0 within numeric tolerance\n";}
        if(a.model.window)cout<<"SELECTED WINDOW     : "<<a.model.window<<" observations\n";
        printValidationSummary(a.model);
        if(!a.model.note.empty())cout<<"MODEL NOTE          : "<<a.model.note<<"\n";
    }else{
        if(!a.warning.empty())cout<<"WARNING             : "<<a.warning<<"\n";
        if(!a.changePoints.empty()){cout<<"CHANGE-POINT CANDIDATES: ";for(size_t i=0;i<a.changePoints.size()&&i<10;++i){if(i)cout<<", ";cout<<a.changePoints[i];}cout<<"\n";}
        cout<<"PREDICTION          : Not released because rolling validation did not establish a stable future path.\n";
    }
    cout<<"NOTE                : Evidence score is a model-selection score, NOT a probability of correctness. Extrapolation can fail outside the observed domain.\n";
    cout<<"=====================================================\n";
    if(!a.leaderboard.empty()){
        cout<<"Top validated candidates:\n";
        const int show=min(10,sizeToInt(a.leaderboard.size()));
        for(int i=0;i<show;++i){
            const auto&m=a.leaderboard[i];
            cout<<"  "<<i+1<<") "<<left<<setw(38)<<m.name<<right
                <<" score="<<fixed<<setprecision(2)<<(double)m.metrics.score<<"%"
                <<" p90Dev="<<fmtFixed(m.metrics.p90DeviationPct,3)<<"%"
                <<" R2="<<fmt(m.metrics.r2,4)<<" window="<<m.window<<"\n";
        }
    }
}

void printMultivariate(const MultiAnalysis&a,int horizon){
    cout<<"\n================ MULTIVARIATE FORECAST ENGINE =============\n";
    cout<<"Rows / variables       : "<<a.rows<<" / "<<a.variables<<"\n";
    cout<<"Global validation tol  : "<<fmtFixed(a.tolerancePct,3)<<"%\n";
    cout<<"Primary approach       : joint regularized lag model (VAR-style), rolling-origin validation, with chained fallback\n";
    if(a.fullForecastAvailable){
        if(!a.links.empty()){
            const auto&r=a.links.front();
            cout<<"MODEL STATUS           : "<<(r.found?"ACCEPTED":"REJECTED")<<"\n";
            cout<<"JOINT MODEL            : "<<r.category<<"\n";
            cout<<"LOGIC                  : "<<r.logic<<"\n";
            cout<<"FORMULA                : "<<r.formula<<"\n";
            cout<<"EVIDENCE SCORE        : "<<fmtFixed(r.evidenceScore,3)<<"%\n";
            if(r.model.metrics.nValid)printValidationSummary(r.model);
        }
        cout<<"\n================ PREDICTED FUTURE ROWS ====================\n";
        for(int h=0;h<horizon&&h<(int)a.predictions.size();++h){
            cout<<"Row +"<<h+1<<" : (";
            for(int j=0;j<a.variables;++j){if(j)cout<<", ";cout<<fmt(a.predictions[h][j]);}
            cout<<")\n";
        }
    }else{
        cout<<"CHAIN / JOINT STATUS   : STOPPED\n";
        cout<<"REASON                 : "<<a.failureReason<<"\n";
    }
    cout<<"===========================================================\n";
}

// ----------------------------- Interactive UI -------------------
string readLinePrompt(const string& prompt) {
    cout << prompt << std::flush;
    string line;
    if (!getline(cin, line)) return {};
    return line;
}

bool parseIntStrict(const string& s, int& out) {
    string t=s;
    size_t a=t.find_first_not_of(" \t\r\n"), b=t.find_last_not_of(" \t\r\n");
    if(a==string::npos)return false;
    t=t.substr(a,b-a+1);
    try {
        size_t pos=0; long long v=stoll(t,&pos);
        if(pos!=t.size() || v<INT_MIN || v>INT_MAX)return false;
        out=(int)v; return true;
    } catch(...) { return false; }
}

int askIntRange(const string& prompt, int lo, int hi, int def) {
    while (true) {
        cout << prompt << std::flush;
        string line;
        if (!getline(cin, line)) {
            g_inputClosed = true;
            return (def >= lo && def <= hi) ? def : lo;
        }
        const bool blank = line.find_first_not_of(" \t\r\n") == string::npos;
        if (blank && def >= lo && def <= hi) return def;
        int value = 0;
        if (parseIntStrict(line, value) && value >= lo && value <= hi) return value;
        cout << "  Please enter a number from " << lo << " to " << hi << ".\n";
    }
}

bool askYesNo(const string& prompt, bool def) {
    while(true){
        cout<<prompt<<(def?" [Y/n]: ":" [y/N]: ") << std::flush;
        string s; if(!getline(cin,s)){ g_inputClosed=true; return def; }
        if(s.empty())return def;
        char c=(char)tolower((unsigned char)s[0]);
        if(c=='y') return true;
        if(c=='n') return false;
        cout<<"  Please answer y or n.\n";
    }
}

ForecastOptions askForecastOptions(bool showPlot=true) {
    ForecastOptions o;
    o.horizon=askIntRange("How many next terms/rows do you want to predict? [1]: ",1,1000,1);
    cout << "\nTolerance mode:\n"
         << "1) Automatic (dataset-dependent)\n"
         << "2) Custom maximum deviation percentage\n";
    const int tm=askIntRange("> ",1,2,1);
    if(tm==2){
        while(true){
            cout<<"Enter maximum accepted deviation in % (example: 0.3 means 0.3%): "<<std::flush;
            string s; if(!getline(cin,s)){ g_inputClosed=true; break; }
            try{
                size_t p=0; ld v=stold(s,&p);
                while(p<s.size() && isspace((unsigned char)s[p])) ++p;
                if(p==s.size() && isfinite(v) && v>=0 && v<=100){
                    o.customTolerance=true;o.tolerancePct=v;
                    cout<<"Accepted custom tolerance: "<<fmtFixed(v,3)<<"%\n";
                    break;
                }
            }catch(...){ }
            cout<<"  Enter a value from 0 to 100. Example: 0.3 = 0.3%.\n";
        }
    }else{
        cout<<"Using automatic dataset-dependent tolerance.\n";
    }
    if(showPlot){
        cout<<"\n";
        o.requestPlot=askYesNo("Generate a plot after analysis?",false);
    }
    return o;
}

void printMethodCard(int method) {
    cout << "\n========================================================\n";
    switch(method){
        case 1:
            cout << "1) SINGLE 1D SEQUENCE\n"
                 << "   Enter one sequence of numeric observations.\n"
                 << "   The engine tests exact rules first, then validated\n"
                 << "   regression/recurrence/composite models and can forecast\n"
                 << "   multiple future terms.\n";
            break;
        case 2:
            cout << "2) N-DIMENSIONAL / MULTIVARIATE DATASET\n"
                 << "   Enter rows such as (4,5,7) or 4 5 7.\n"
                 << "   You choose the number of variables first.\n"
                 << "   Forecasting uses a joint VAR-style ridge model first, with\n"
                 << "   chained fallback when a joint model is not stable.\n";
            break;
        case 3:
            cout << "3) MANUAL COORDINATES\n"
                 << "   2D Cartesian, 2D polar, 3D Cartesian, cylindrical,\n"
                 << "   or spherical coordinates can be entered interactively.\n";
            break;
        case 4:
            cout << "4) LOAD DATASET / SEQUENCE FROM FILE\n"
                 << "   Choose the file interpretation, enter its path, then\n"
                 << "   the engine analyzes the loaded observations.\n";
            break;
        case 5:
            cout << "5) CRYPTO / ENCODING LAB\n"
                 << "   Classical ciphers, Morse, Base64/hex, SHA-256, HMAC,\n"
                 << "   PBKDF2 key derivation, optional AES-256-GCM, and analysis.\n";
            break;
        case 6:
            cout << "6) STATISTICAL ANALYSIS ONLY\n"
                 << "   Descriptive statistics and distribution diagnostics\n"
                 << "   without attempting to predict future terms.\n";
            break;
        case 7:
            cout << "7) RANDOM / PRNG TOOLS\n"
                 << "   Analyze an existing random/PRNG-like series or\n"
                 << "   generate a reproducible series with several generators.\n";
            break;
        case 8:
            cout << "8) GRAPH / CURVE ANALYSIS\n"
                 << "   Enter x-y graph points, inspect slope/turning behavior,\n"
                 << "   then run the same validated extrapolation engine.\n";
            break;
        case 9:
            cout << "9) ENGINE SELF-TEST / BENCHMARK\n"
                 << "   Verify cryptographic primitives and core forecasting\n"
                 << "   behavior against known/reference cases.\n";
            break;
        case 10:
            cout << "10) GENERIC PUZZLE / CUBE GRAPH SOLVER\n"
                 << "   IDA* over explicit state transitions; includes a generated NxNxN sticker-cube move set.\n";
            break;
        default:
            cout << "Unknown method.\n";
            break;
    }
    cout << "========================================================\n" << std::flush;
}

vector<ld> readSequenceInteractive(){
    g_lastInputCancelled = false;
    cout << "\nDATA ENTRY\n"
         << "Enter numeric observations in any of these forms:\n"
         << "  1 2 3 4 5\n"
         << "  1, 2, 3, 4, 5\n"
         << "  1\n  2\n  3\n  4\n"
         << "You may mix both styles. A blank line finishes the dataset.\n"
         << "Malformed/mixed text is rejected as a whole line.\n"
         << "Type 'help' for examples or 'back' to return to the main menu.\n"
         << "--------------------------------------------------------\n";

    vector<ld> nums;
    while (true) {
        cout << "data> " << std::flush;
        string line;
        if (!getline(cin, line)) {
            g_inputClosed = true;
            break;
        }
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == string::npos) break;
        const size_t last = line.find_last_not_of(" \t\r\n");
        const string trimmed = line.substr(first, last - first + 1);
        if (trimmed == "help" || trimmed == "HELP" || trimmed == "?") {
            cout << "  Examples: 2 4 6 8 | 2,4,6,8 | one value per line.\n"
                 << "  Parentheses/brackets are also accepted.\n"
                 << "  Blank line = finish, 'back' = cancel this dataset.\n";
            continue;
        }
        if (trimmed == "back" || trimmed == "BACK" || trimmed == "cancel" || trimmed == "CANCEL") {
            nums.clear();
            g_lastInputCancelled = true;
            break;
        }
        const auto values = parseNumberLine(line);
        if (values.empty()) {
            cout << "  No numeric values found on that line. Try again or type 'help'.\n";
            continue;
        }
        nums.insert(nums.end(), values.begin(), values.end());
        cout << "  Accepted " << values.size() << " value" << (values.size() == 1 ? "" : "s")
             << ". Total observations: " << nums.size() << "\n";
    }
    if (nums.size() < 3 && !nums.empty()) {
        cout << "  Need at least 3 observations; dataset was not analyzed.\n";
        nums.clear();
    }
    return nums;
}

// --------------------------- Generic puzzle / graph search ---------------------------
struct PuzzleMove {
    string name;
    vector<int> perm; // next[i] = current[perm[i]]
    int group=-1;
};

vector<int> applyMove(const vector<int>& state,const PuzzleMove& m){
    vector<int> out(state.size());for(size_t i=0;i<state.size();++i)out[i]=state[m.perm[i]];return out;
}

string stateKey(const vector<int>& s){
    string k; k.reserve(s.size()*3);
    for(int v:s){k.append(to_string(v));k.push_back(',');}
    return k;
}

bool sameMultiset(const vector<int>&a,const vector<int>&b){
    if(a.size()!=b.size()) return false;
    unordered_map<int,int> c;
    for(int v:a) ++c[v];
    for(int v:b) if(--c[v]<0) return false;
    for(auto&q:c) if(q.second) return false;
    return true;
}

int countMisplaced(const vector<int>&s,const vector<int>&g){int c=0;for(size_t i=0;i<s.size();++i)if(s[i]!=g[i])++c;return c;}

struct IDAStarSolver {
    vector<PuzzleMove> moves;
    vector<int> goal,path;
    unordered_map<string,int> bestDepth;
    int maxAffected=1,nodes=0,maxNodes=2000000;
    bool solved=false;
    int heuristic(const vector<int>&s)const{return (countMisplaced(s,goal)+maxAffected-1)/maxAffected;}
    int dfs(const vector<int>&s,int g,int bound,int prevGroup){
        ++nodes;if(nodes>maxNodes)return INT_MAX/4;
        int h=heuristic(s),f=g+h;if(f>bound)return f;if(s==goal){solved=true;return g;}
        string key=stateKey(s);auto it=bestDepth.find(key);if(it!=bestDepth.end()&&it->second<=g)return INT_MAX/4;bestDepth[key]=g;
        int nextBound=INT_MAX/4;
        vector<int>order(moves.size());iota(order.begin(),order.end(),0);
        stable_sort(order.begin(),order.end(),[&](int a,int b){return countMisplaced(applyMove(s,moves[a]),goal)<countMisplaced(applyMove(s,moves[b]),goal);});
        for(int mi:order){if(moves[mi].group==prevGroup)continue;vector<int>ns=applyMove(s,moves[mi]);path.push_back(mi);int r=dfs(ns,g+1,bound,moves[mi].group);if(solved)return r;path.pop_back();if(r<nextBound)nextBound=r;if(nodes>maxNodes)return INT_MAX/4;}
        return nextBound;
    }
    vector<int> solve(const vector<int>&start,const vector<int>&g,int maxDepth=10,int nodeLimit=2000000){
        if(start.size()!=g.size()||!sameMultiset(start,g)) return{};
        goal=g; maxNodes=nodeLimit; nodes=0; solved=false; path.clear(); maxAffected=1;
        for(auto&m:moves){int a=0;for(size_t i=0;i<m.perm.size();++i)if(m.perm[i]!=(int)i)++a;maxAffected=max(maxAffected,a);}
        bestDepth.clear();
        int h=heuristic(start);for(int bound=h;bound<=maxDepth;++bound){bestDepth.clear();int r=dfs(start,0,bound,-1);if(solved)return path;if(r==INT_MAX/4)break;}return{};
    }
};

struct CubeSticker { int x=0,y=0,z=0; int nx=0,ny=0,nz=0; };

string stickerKey(const CubeSticker&s){return to_string(s.x)+":"+to_string(s.y)+":"+to_string(s.z)+":"+to_string(s.nx)+":"+to_string(s.ny)+":"+to_string(s.nz);}

vector<CubeSticker> makeCubeStickers(int N){
    vector<CubeSticker>v;int s=N-1;
    auto addFace=[&](int face){for(int r=0;r<N;++r)for(int c=0;c<N;++c){int x=0,y=0,z=0,nx=0,ny=0,nz=0;\
        if(face==0){x=2*c-s;y=s;z=s-2*r;ny=1;}\
        if(face==1){x=s;y=s-2*r;z=2*c-s;nx=1;}\
        if(face==2){x=2*c-s;y=s-2*r;z=s;nz=1;}\
        if(face==3){x=2*c-s;y=-s;z=2*r-s;ny=-1;}\
        if(face==4){x=-s;y=s-2*r;z=s-2*c;nx=-1;}\
        if(face==5){x=s-2*c;y=s-2*r;z=-s;nz=-1;}\
        v.push_back({x,y,z,nx,ny,nz});}};
    for(int f=0;f<6;++f) addFace(f);
    return v;
}

CubeSticker rotateSticker(CubeSticker q,int axis,int signedAngle){
    // signedAngle is +/-1 for +/-90 degrees around the positive coordinate axis.
    auto rot=[&](int a,int b,int&oa,int&ob){if(signedAngle>0){oa=-b;ob=a;}else{oa=b;ob=-a;}};
    if(axis==0){int oa,ob;rot(q.y,q.z,oa,ob);q.y=oa;q.z=ob;rot(q.ny,q.nz,oa,ob);q.ny=oa;q.nz=ob;}
    if(axis==1){int oa,ob;rot(q.z,q.x,oa,ob);q.z=oa;q.x=ob;rot(q.nz,q.nx,oa,ob);q.nz=oa;q.nx=ob;}
    if(axis==2){int oa,ob;rot(q.x,q.y,oa,ob);q.x=oa;q.y=ob;rot(q.nx,q.ny,oa,ob);q.nx=oa;q.ny=ob;}
    return q;
}

vector<int> cubeFacePermutation(int N,int face,int turn){
    auto stickers=makeCubeStickers(N);unordered_map<string,int>index;for(int i=0;i<(int)stickers.size();++i)index[stickerKey(stickers[i])]=i;
    const int s=N-1;int axis=(face==1||face==4)?0:(face==0||face==3)?1:2;int sign=(face==0||face==1||face==2)?1:-1;int angle=turn*sign*-1; // clockwise as viewed from the outward face
    vector<int>perm(stickers.size());iota(perm.begin(),perm.end(),0);
    for(int i=0;i<(int)stickers.size();++i){auto q=stickers[i];int layer=(axis==0?q.x:(axis==1?q.y:q.z));int target=(axis==0?q.nx:(axis==1?q.ny:q.nz))*s;if(layer==target){auto r=rotateSticker(q,axis,angle);auto it=index.find(stickerKey(r));if(it==index.end())return{};perm[it->second]=i;}}
    return perm;
}

vector<PuzzleMove> makeCubeMoves(int N){
    vector<PuzzleMove>m;const string faces="URFDLB";for(int f=0;f<6;++f){auto cw=cubeFacePermutation(N,f,1);if(cw.empty())return{};auto ccw=cubeFacePermutation(N,f,-1);if(ccw.empty())return{};vector<int>dbl=cw;dbl= [&](){vector<int>v(dbl.size());for(size_t i=0;i<dbl.size();++i)v[i]=dbl[cw[i]];return v;}();m.push_back({string(1,faces[f]),cw,f});m.push_back({string(1,faces[f])+"'",ccw,f});m.push_back({string(1,faces[f])+"2",dbl,f});}return m;
}

int cubeMoveIndex(const vector<PuzzleMove>&m,const string&t){for(int i=0;i<(int)m.size();++i)if(m[i].name==t)return i;return-1;}

vector<string> tokenizeMoves(const string&s){vector<string>v;istringstream in(s);string q;while(in>>q){if(q.size()==2&&(q[1]=='\''))v.push_back(q);else if(q.size()>=2&&q[1]=='2')v.push_back(q.substr(0,1)+"2");else v.push_back(q.substr(0,1));}return v;}

void runPuzzleMode(){
    printMethodCard(10);
    cout<<"1) Generic permutation/state puzzle\n2) NxNxN sticker cube\n> "<<flush;
    int kind=askIntRange("",1,2,1);
    if(kind==2){
        int N=askIntRange("Cube size N (2-4): ",2,4,3);
        auto moves=makeCubeMoves(N);if(moves.empty()){cout<<"Cube move construction failed.\n";return;}
        auto solved=[&](){vector<int>s(6*N*N);for(int f=0;f<6;++f)for(int i=0;i<N*N;++i)s[f*N*N+i]=f;return s;}();
        string scramble=readLinePrompt("Scramble (e.g. R U R' U'): ");
        vector<int>state=solved;for(const string&t:tokenizeMoves(scramble)){int mi=cubeMoveIndex(moves,t);if(mi<0){cout<<"Unknown move: "<<t<<"\n";return;}state=applyMove(state,moves[mi]);}
        int depth=askIntRange("Maximum search depth [8]: ",0,16,8);
        IDAStarSolver solver;solver.moves=moves;auto path=solver.solve(state,solved,depth,3000000);
        if(path.empty()){cout<<"No solution within depth/node limit. Nodes="<<solver.nodes<<"\n";return;}
        cout<<"Solution: ";for(size_t i=0;i<path.size();++i){if(i)cout<<' ';cout<<moves[path[i]].name;}cout<<"\nNodes: "<<solver.nodes<<"\n";
        cout<<"Note: this is a generic IDA* solver; it is not a Kociemba/two-phase speed solver, so deep 3x3 scrambles can exceed the search limit.\n";
        return;
    }
    int n=askIntRange("Number of state positions: ",2,128,8);
    auto parseN=[&](const string&prompt){while(true){auto v=parseNumberLine(readLinePrompt(prompt));if((int)v.size()!=n){cout<<"Need exactly "<<n<<" integers.\n";continue;}vector<int>q;bool ok=true;for(ld z:v)if(!isInteger(z)||z<-1000000||z>1000000){ok=false;break;}if(!ok){cout<<"Invalid integer state.\n";continue;}for(ld z:v)q.push_back((int)llroundl(z));return q;}};
    vector<int>start=parseN("Start state: ");vector<int>goal=parseN("Goal state : ");
    int mc=askIntRange("Number of moves: ",1,64,4);vector<PuzzleMove>moves;
    for(int i=0;i<mc;++i){string name=readLinePrompt("Move "+to_string(i+1)+" name: ");auto pv=parseNumberLine(readLinePrompt("Permutation (new[i] = old[perm[i]]): "));if((int)pv.size()!=n){cout<<"Bad permutation size.\n";return;}vector<int>p(n);vector<int>seen(n);for(int j=0;j<n;++j){if(!isInteger(pv[j])||pv[j]<0||pv[j]>=n||seen[(int)pv[j]]){cout<<"Invalid permutation.\n";return;}p[j]=(int)pv[j];seen[p[j]]=1;}moves.push_back({name,p,i});}
    int depth=askIntRange("Maximum search depth [10]: ",0,24,10);IDAStarSolver solver;solver.moves=moves;auto path=solver.solve(start,goal,depth,3000000);if(path.empty())cout<<"No solution within depth/node limit. Nodes="<<solver.nodes<<"\n";else{cout<<"Solution: ";for(size_t i=0;i<path.size();++i){if(i)cout<<' ';cout<<moves[path[i]].name;}cout<<"\nNodes: "<<solver.nodes<<"\n";}
}

void printMainMenu(){
    cout<<"\n========================================================\n"
         <<" ADVANCED MULTIVARIATE SEQUENCE / CURVE / CRYPTO ENGINE v6.0\n"
         <<"========================================================\n"
         <<"Core input methods:\n"
         <<"1) Single 1D Sequence\n"
         <<"2) N-Dimensional / Multivariate Dataset\n"
         <<"3) Manual Coordinates (2D/3D)\n"
         <<"4) Load Dataset / Sequence from File\n"
         <<"5) Crypto / encoding / classical + modern primitives\n"
         <<"\nAdditional tools (original functionality retained):\n"
         <<"6) Statistical analysis only\n"
         <<"7) Random / PRNG analysis and generation\n"
         <<"8) Graph / curve analysis (x,y input)\n"
         <<"9) Engine self-test / benchmark\n"
         <<"10) Generic puzzle / cube graph solver\n"
         <<"0) Exit\n"
         <<"\nThe screen is interactive: choose a number, then follow the guided prompts.\n"
         <<"v6.0 adds root-aware fitting, numerical roots/turning points for arbitrary fitted models, deep analytic transforms,\n"
         <<"signal segmentation, chirp/recurrence diagnostics, sign-mirror validation, and generic puzzle graph search.\n"
         <<"========================================================\n" << std::flush;
}

void runStatsMode(){
    cout<<"\n========================================================\n"
         <<"6) STATISTICAL ANALYSIS ONLY\n"
         <<"   Descriptive statistics, distribution diagnostics,\n"
         <<"   autocorrelation, runs test, Bernoulli/Poisson checks,\n"
         <<"   confidence interval and uniform-bin diagnostic.\n"
         <<"========================================================\n";
    cout<<"Statistics-only mode does not require a prediction horizon or tolerance.\n";
    vector<ld>nums=readSequenceInteractive();
    if(nums.empty()){
        if(g_lastInputCancelled) { cout << "Input cancelled. Returning to the main menu.\n"; return; }
        cout<<"No numeric observations supplied.\n";return;
    }
    printStats(nums,"Input");
}

void runRandomMode(){
    cout<<"\n================ RANDOM / PRNG TOOLS =====================\n"
         <<"1) Analyze an existing random/PRNG-like series\n"
         <<"2) Generate a random/PRNG series\n"
         <<"0) Back\n"<<std::flush;
    int sub=askIntRange("> ",0,2,0);
    if(sub==1){
        printMethodCard(7);
        cout << "\nRANDOM DATA ENTRY\n"
             << "Enter at least 6 numeric values. One per line or many per line are both accepted.\n";
        const auto nums = readSequenceInteractive();
        if (!nums.empty()) printRandomDiagnostics(nums);
        else if(g_lastInputCancelled) cout << "Input cancelled. Returning to the PRNG menu.\n";
    } else if(sub==2) generateRandomSeries();
}

void runSingleSequence(){
    printMethodCard(1);
    vector<ld> nums=readSequenceInteractive();
    if(g_lastInputCancelled) { cout << "Input cancelled. Returning to the main menu.\n"; return; }
    if(nums.size()<3){cout<<"Need at least 3 observations.\n";return;}
    ForecastOptions o=askForecastOptions(true);
    vector<Point>pts;for(size_t i=0;i<nums.size();++i)pts.emplace_back(Point{static_cast<ld>(i)+1,nums[i],0});
    vector<ld>x(nums.size());iota(x.begin(),x.end(),1);
    Analysis res=analyzeSequence(x,nums,o.horizon,o.customTolerance?o.tolerancePct:-1.0L);
    printAnalysis(res,o.horizon);
    if(!res.found)printCompactFallbackDiagnostics(nums,res.warning.empty()?"no validated model":res.warning,o.customTolerance?o.tolerancePct:-1.0L);
    else if(o.requestPlot) {writePlotFiles(pts,res.model.valid?&res.model:nullptr,res.model.valid?res.model.formula:"No validated model",false,o.horizon);asciiPlot(pts,res.model.valid?&res.model:nullptr);}
}

void runMultivariate(){
    printMethodCard(2);
    const int variables = askIntRange("How many variables are in each row? (2-64): ", 2, 64, -1);
    if (g_inputClosed) return;

    cout << "\nRow entry mode:\n"
         << "1) Row-by-row (recommended): one complete row per line\n"
         << "2) Flexible stream: values can be pasted in any line grouping\n"
         << "Both modes accept commas, spaces, parentheses and brackets.\n";
    const int style = askIntRange("Choose entry mode (1-2): ", 1, 2, 2);
    if (g_inputClosed) return;

    cout << "\nDATA ENTRY\n"
         << "Expected row width: " << variables << " values.\n"
         << "Example row: (4, 8, 12)\n"
         << "Blank line finishes input. Type 'help' for examples or 'back' to cancel.\n"
         << "--------------------------------------------------------\n";

    vector<vector<ld>> rows;
    vector<ld> pending;
    g_lastInputCancelled = false;
    while (true) {
        cout << "row> " << std::flush;
        string line;
        if (!getline(cin, line)) { g_inputClosed = true; break; }
        const size_t first = line.find_first_not_of(" \t\r\n");
        if (first == string::npos) break;
        const size_t last = line.find_last_not_of(" \t\r\n");
        const string trimmed = line.substr(first, last - first + 1);
        if (trimmed == "help" || trimmed == "HELP" || trimmed == "?") {
            cout << "  Row-by-row: 4 8 12\n"
                 << "  Flexible stream: 4 8 12 5 10 15 6 12 18\n"
                 << "  Flexible mode also accepts one value per line.\n";
            continue;
        }
        if (trimmed == "back" || trimmed == "BACK" || trimmed == "cancel" || trimmed == "CANCEL") {
            rows.clear();
            pending.clear();
            g_lastInputCancelled = true;
            break;
        }
        const auto values = parseNumberLine(line);
        if (values.empty()) {
            cout << "  No numeric values found. Try again or type 'help'.\n";
            continue;
        }

        if (style == 1 && values.size() == static_cast<size_t>(variables) && pending.empty()) {
            rows.emplace_back(values);
        } else {
            pending.insert(pending.end(), values.begin(), values.end());
            while (pending.size() >= static_cast<size_t>(variables)) {
                rows.emplace_back(pending.begin(), pending.begin() + variables);
                pending.erase(pending.begin(), pending.begin() + variables);
            }
        }
        cout << "  Complete rows: " << rows.size();
        if (!pending.empty()) {
            const size_t need = static_cast<size_t>(variables) - pending.size();
            cout << " | waiting for " << need << " more value" << (need == 1 ? "" : "s");
        }
        cout << "\n";
    }

    if (g_lastInputCancelled) { cout << "Input cancelled. Returning to the main menu.\n"; return; }
    if (!pending.empty())
        cout << "  Ignoring " << pending.size() << " incomplete trailing value" << (pending.size() == 1 ? "" : "s") << ".\n";
    if (rows.size() < 8) { cout << "Need at least 8 complete rows for multivariate validation.\n"; return; }

    ForecastOptions o = askForecastOptions(false);
    if (g_inputClosed) return;
    o.requestPlot = askYesNo("Generate a plot of the multivariate driver/forecast?", false);
    if (g_inputClosed) return;
    MultiAnalysis ma = analyzeMultivariate(rows, o.horizon, o.customTolerance ? o.tolerancePct : -1.0L);
    printMultivariate(ma, o.horizon);
    if (!ma.fullForecastAvailable) {
        vector<ld> v; v.reserve(rows.size());
        for (const auto& row : rows) v.push_back(row[0]);
        printCompactFallbackDiagnostics(v, ma.failureReason, o.customTolerance?o.tolerancePct:-1.0L);
    } else if (o.requestPlot) {
        vector<Point> pts; pts.reserve(rows.size());
        vector<ld> x(rows.size()); iota(x.begin(), x.end(), 1);
        for (size_t i = 0; i < rows.size(); ++i) pts.emplace_back(Point{x[i], rows[i][0], 0});
        writeMultivariatePlotFiles(rows, ma.predictions);
        writePlotFiles(pts, ma.driver.model.valid ? &ma.driver.model : nullptr, "Multivariate driver / chained forecast", false, o.horizon);
        asciiPlot(pts, ma.driver.model.valid ? &ma.driver.model : nullptr);
        cout << "Plot output: multivariate_analysis.png (plus driver ASCII plot).\n";
    }
}

void runCoordinates(){
    printMethodCard(3);
    cout<<"Coordinate representation:\n"
         <<"1) 2D Cartesian (x y)\n"
         <<"2) 2D Polar (r theta)\n"
         <<"3) 3D Cartesian (x y z)\n"
         <<"4) 3D Cylindrical (r theta z)\n"
         <<"5) 3D Spherical (r theta phi)\n";
    int mode=askIntRange("> ",1,5,1);
    bool rad=true;
    if(mode==2||mode==4||mode==5){int u=askIntRange("Angle units: 1) radians  2) degrees\n> ",1,2,1);rad=(u==1);}
    cout<<"Enter points one per line; blank line ends input.\n"<<std::flush;
    vector<Point>pts;string line;bool is3D=(mode>=3);
    while(true){if(!getline(cin,line))break;if(line.find_first_not_of(" \t\r\n")==string::npos)break;auto r=parseNumberLine(line);bool ok=false;
        if(mode==1&&r.size()>=2){pts.emplace_back(Point{r[0],r[1],0});ok=true;}
        else if(mode==2&&r.size()>=2){ld a=rad?r[1]:r[1]*PI/180;pts.emplace_back(Point{r[0]*cosl(a),r[0]*sinl(a),0});ok=true;}
        else if(mode==3&&r.size()>=3){pts.emplace_back(Point{r[0],r[1],r[2]});ok=true;}
        else if(mode==4&&r.size()>=3){ld a=rad?r[1]:r[1]*PI/180;pts.emplace_back(Point{r[0]*cosl(a),r[0]*sinl(a),r[2]});ok=true;}
        else if(mode==5&&r.size()>=3){ld th=rad?r[1]:r[1]*PI/180,ph=rad?r[2]:r[2]*PI/180;pts.emplace_back(Point{r[0]*sinl(ph)*cosl(th),r[0]*sinl(ph)*sinl(th),r[0]*cosl(ph)});ok=true;}
        if(!ok)cout<<"  Ignored malformed point.\n";
    }
    if(pts.size()<3){cout<<"Need at least 3 points.\n";return;}
    cout<<"\nFor coordinate prediction, choose which coordinate becomes the analyzed response:\n";
    int response=askIntRange(is3D?"1) X\n2) Y\n3) Z\n> ":"1) X\n2) Y\n> ",1,is3D?3:2,1)-1;
    vector<ld>nums;nums.reserve(pts.size());vector<ld>x; x.reserve(pts.size());
    for(size_t i=0;i<pts.size();++i){x.push_back(mode==1?pts[i].x:(ld)i+1);nums.push_back(response==0?pts[i].x:(response==1?pts[i].y:pts[i].z));}
    ForecastOptions o=askForecastOptions(true);
    Analysis res=analyzeSequence(x,nums,o.horizon,o.customTolerance?o.tolerancePct:-1.0L);printAnalysis(res,o.horizon);
    if(!res.found)printCompactFallbackDiagnostics(nums,res.warning,o.customTolerance?o.tolerancePct:-1.0L);
    else if(o.requestPlot){writePlotFiles(pts,res.model.valid?&res.model:nullptr,res.model.valid?res.model.formula:"No validated model",is3D,o.horizon);asciiPlot(pts,res.model.valid?&res.model:nullptr);}
}


void printCurveDiagnostics(const vector<Point>& pts){
    if(pts.size()<3)return;
    vector<ld>slope; slope.reserve(pts.size()-1);
    for(size_t i=1;i<pts.size();++i){ld dx=pts[i].x-pts[i-1].x; slope.push_back(fabsl(dx)<1e-18L?numeric_limits<ld>::quiet_NaN():(pts[i].y-pts[i-1].y)/dx);}
    bool inc=true,dec=true;for(ld q:slope){if(!isfinite(q))continue;if(q<-1e-9L)inc=false;if(q>1e-9L)dec=false;}
    cout<<"\n================ CURVE / GRAPH DIAGNOSTICS ================\n";
    cout<<"Points              : "<<pts.size()<<"\n";
    cout<<"X range             : "<<fmt(pts.front().x)<<" .. "<<fmt(pts.back().x)<<"\n";
    cout<<"Y range             : ";ld ymin=pts.front().y,ymax=pts.front().y;for(auto&p:pts){ymin=min(ymin,p.y);ymax=max(ymax,p.y);}cout<<fmt(ymin)<<" .. "<<fmt(ymax)<<"\n";
    cout<<"Monotonicity        : ";if(inc&&!dec)cout<<"increasing";else if(dec&&!inc)cout<<"decreasing";else cout<<"non-monotonic";cout<<"\n";
    int turns=0;for(size_t i=1;i<slope.size();++i)if(isfinite(slope[i-1])&&isfinite(slope[i])&&((slope[i-1]>0&&slope[i]<0)||(slope[i-1]<0&&slope[i]>0)))++turns;
    cout<<"Estimated turning points: "<<turns<<"\n";
    if(!slope.empty()){cout<<"Approx. end slope    : "<<fmt(slope.back())<<"\n";}
    cout<<"===========================================================\n";
}

void runGraphCurveMode(){
    printMethodCard(8);
    cout<<"Enter x y pairs, one pair per line. Blank line finishes.\n";
    vector<Point>pts; string line;
    while(getline(cin,line)){
        if(line.find_first_not_of(" \\t\\r\\n")==string::npos)break;
        auto v=parseNumberLine(line);
        if(v.size()!=2){cout<<"  Need exactly two numeric values: x y.\n";continue;}
        pts.push_back({v[0],v[1],0});
    }
    if(pts.size()<5){cout<<"Need at least 5 graph points.\n";return;}
    sort(pts.begin(),pts.end(),[](const Point&a,const Point&b){return a.x<b.x;});
    bool uniqueX=true;for(size_t i=1;i<pts.size();++i)if(fabsl(pts[i].x-pts[i-1].x)<1e-15L)uniqueX=false;
    if(!uniqueX){cout<<"Duplicate x values are not supported in curve regression mode.\n";return;}
    printCurveDiagnostics(pts);
    vector<ld>x,y;for(auto&p:pts){x.push_back(p.x);y.push_back(p.y);}ForecastOptions o=askForecastOptions(true);
    cout<<"Forecast x-step [1.0]: "<<flush;string ds;getline(cin,ds);ld dx=1;try{if(!ds.empty())dx=stold(ds);}catch(...){dx=1;}
    Analysis a=analyzeSequence(x,y,o.horizon,o.customTolerance?o.tolerancePct:-1.0L);
    // The generic sequence engine assumes x+1 for forecasting; re-fit the selected family at
    // final x and project using the requested graph spacing so curve inputs remain true x/y data.
    if(a.found && a.model.predict){
        a.predictions.clear();
        for(int h=1;h<=o.horizon;++h)a.predictions.push_back(a.model.predict(x.back()+dx*h));
        a.nextValue=a.predictions.empty()?numeric_limits<ld>::quiet_NaN():a.predictions.front();
        auto half=empiricalForecastBand(a.predictions,a.model.validationResiduals,0.90L);
        a.predictionLow.clear(); a.predictionHigh.clear();
        if(half.size()==a.predictions.size()){
            a.predictionLow.resize(a.predictions.size()); a.predictionHigh.resize(a.predictions.size());
            for(size_t i=0;i<a.predictions.size();++i){a.predictionLow[i]=a.predictions[i]-half[i];a.predictionHigh[i]=a.predictions[i]+half[i];}
        }
        a.model.nextValue=a.nextValue;
    }
    printAnalysis(a,o.horizon);
    if(a.found&&o.requestPlot)writePlotFiles(pts,&a.model,a.model.formula,false,o.horizon,dx);
}

string fileExtensionLower(const string&fn){
    auto p=fn.find_last_of(".");if(p==string::npos)return{};string e=fn.substr(p+1);transform(e.begin(),e.end(),e.begin(),[](unsigned char c){return (char)tolower(c);});return e;
}

vector<string> splitCSVFields(const string&line,char delimiter){
    vector<string>out;string cur;bool quoted=false;
    for(size_t i=0;i<line.size();++i){char c=line[i];if(c=='"'){if(quoted&&i+1<line.size()&&line[i+1]=='"'){cur.push_back('"');++i;}else quoted=!quoted;}else if(c==delimiter&&!quoted){out.push_back(cur);cur.clear();}else cur.push_back(c);}out.push_back(cur);return out;
}

bool loadTextRows(const string&path,vector<vector<ld>>&rows,string&note){
    ifstream fin(path);if(!fin){note="Cannot open file.";return false;}string line;rows.clear();char delim=0;bool headerSkipped=false;
    while(getline(fin,line)){
        if(line.size()>=3&&(unsigned char)line[0]==0xEF&&(unsigned char)line[1]==0xBB&&(unsigned char)line[2]==0xBF)line.erase(0,3);
        if(line.find_first_not_of(" \\t\\r\\n")==string::npos)continue;
        if(!delim){if(line.find('\t')!=string::npos)delim='\t';else if(line.find(',')!=string::npos)delim=',';else if(line.find(';')!=string::npos)delim=';';else delim=' ';}
        vector<ld>vals;
        if(delim==' '){vals=parseNumberLine(line);}else{
            for(string f:splitCSVFields(line,delim)){auto v=parseNumberLine(f);if(v.size()!=1){vals.clear();break;}vals.push_back(v[0]);}
        }
        if(vals.empty()){
            if(!headerSkipped && rows.empty()){headerSkipped=true;continue;}
            continue;
        }
        rows.push_back(vals);
    }
    if(rows.empty()){note="No numeric rows were found.";return false;}
    note="Loaded "+to_string(rows.size())+" numeric rows.";return true;
}

bool runPythonSpreadsheetLoader(const string&path,vector<vector<ld>>&rows,string&note){
    string py=preferredPython();
    if(py.empty()){note="Python is not available for Excel/ODS import.";return false;}
    const string ext=fileExtensionLower(path);
    filesystem::path tmpData=filesystem::temp_directory_path()/("sequence_engine_excel_"+to_string(chrono::high_resolution_clock::now().time_since_epoch().count())+".tsv");
    filesystem::path tmpScript=filesystem::temp_directory_path()/("sequence_engine_excel_loader_"+to_string(chrono::high_resolution_clock::now().time_since_epoch().count())+".py");
    ofstream script(tmpScript);
    if(!script){note="Could not create temporary spreadsheet loader.";return false;}
    if(ext=="xlsx"||ext=="xlsm"||ext=="xlx"){
        script<<R"PY(import csv,sys,zipfile,openpyxl,io
p=sys.argv[1]; out=sys.argv[2]
blob=open(p,'rb').read()
if not zipfile.is_zipfile(io.BytesIO(blob)): raise RuntimeError("The .xlx upload is not an OOXML/XLSX workbook.")
w=openpyxl.load_workbook(io.BytesIO(blob),data_only=True,read_only=True)
sh=w.active
with open(out,'w',newline='',encoding='utf-8') as f:
    c=csv.writer(f,delimiter='\t')
    for r in sh.iter_rows(values_only=True):
        vals=[x for x in r if isinstance(x,(int,float)) and x==x]
        if vals: c.writerow(vals)
w.close()
)PY";
    }else{
        script<<R"PY(import sys,pandas as pd
df=pd.read_excel(sys.argv[1],header=None)
df.to_csv(sys.argv[2],sep='\t',index=False,header=False)
)PY";
    }
    script.close();
    string cmd=py+" "+shellQuote(tmpScript.string())+" "+shellQuote(path)+" "+shellQuote(tmpData.string());
    int rc=quietSystem(cmd);
    error_code ec;filesystem::remove(tmpScript,ec);
    if(rc!=0){
        filesystem::remove(tmpData,ec);
        // .xls is an especially common legacy format. pandas requires xlrd for
        // binary XLS, so fall back to LibreOffice headless conversion when xlrd
        // is not installed. The converted XLSX is then handled by openpyxl.
        if(ext=="xls" && commandExists("libreoffice")){
            filesystem::path outDir=filesystem::temp_directory_path()/("sequence_engine_xls_"+to_string(chrono::high_resolution_clock::now().time_since_epoch().count()));
            filesystem::create_directories(outDir,ec);
            string convert=string("libreoffice --headless --convert-to xlsx --outdir ")+shellQuote(outDir.string())+" "+shellQuote(path);
            int cr=quietSystem(convert);
            filesystem::path converted=outDir/(filesystem::path(path).stem().string()+".xlsx");
            if(cr==0&&filesystem::exists(converted)){
                vector<vector<ld>>tmpRows;string tmpNote;
                bool ok2=runPythonSpreadsheetLoader(converted.string(),tmpRows,tmpNote);
                filesystem::remove_all(outDir,ec);
                if(ok2){rows.swap(tmpRows);note="Loaded legacy .xls through LibreOffice conversion -> XLSX -> openpyxl.";return true;}
            }else filesystem::remove_all(outDir,ec);
        }
        note="Spreadsheet import failed. XLSX/XLSM uses openpyxl; XLS uses pandas/xlrd with a LibreOffice fallback; ODS uses pandas/odf.";
        return false;
    }
    bool ok=loadTextRows(tmpData.string(),rows,note);filesystem::remove(tmpData,ec);return ok;
}

bool loadDataRowsAuto(const string&path,vector<vector<ld>>&rows,string&note){
    string ext=fileExtensionLower(path);
    if(ext=="xlsx"||ext=="xlsm"||ext=="xls"||ext=="xlx"||ext=="xlsb"||ext=="ods") return runPythonSpreadsheetLoader(path,rows,note);
    return loadTextRows(path,rows,note);
}

int inferFileMode(const vector<vector<ld>>&rows){
    if(rows.empty())return -1;
    size_t maxCols=0,minCols=numeric_limits<size_t>::max();
    bool allOne=true,allTwo=true,allSame=true;
    size_t firstCols=0;
    for(const auto&r:rows){
        if(r.empty())continue;
        if(firstCols==0)firstCols=r.size();
        maxCols=max(maxCols,r.size());minCols=min(minCols,r.size());
        allOne = allOne && r.size()==1;
        allTwo = allTwo && r.size()==2;
        allSame = allSame && r.size()==firstCols;
    }
    // A plain sequence may be wrapped across lines or supplied as one long line.
    // Prefer sequence interpretation for one-column input or a single long row.
    if(allOne || rows.size()==1)return 0;
    if(allTwo && rows.size()>=3)return 2; // x,y curve
    if(allSame && maxCols>=3)return 1; // multivariate rows
    // Mixed-width numeric input is much more commonly a sequence pasted with
    // inconsistent separators/line breaks than a valid multivariate table.
    return 0;
}

void runFileMode(){
    printMethodCard(4);
    cout<<"File path (TXT/CSV/TSV/XLSX/XLSM/XLS/XLSB/XLX/ODS): "<<std::flush;string fn;getline(cin,fn);
    if(fn.empty()){cout<<"No file path supplied.\n";return;}
    vector<vector<ld>> loaded;string loadNote;
    if(!loadDataRowsAuto(fn,loaded,loadNote)){cout<<"Load failed: "<<loadNote<<"\n";return;}
    cout<<loadNote<<"\n";
    int inferred=inferFileMode(loaded);
    cout<<"Detected structure: "<<(inferred==0?"1D sequence":inferred==2?"2D x-y curve":inferred==1?"multivariate rows":"mixed/ambiguous")<<"\n";
    cout<<"Interpretation:\n"
        <<"0) Automatic (recommended)\n"
        <<"1) N-dimensional / multivariate\n"
        <<"2) 2D Cartesian\n"
        <<"3) 2D Polar\n"
        <<"4) 3D Cartesian\n"
        <<"5) 3D Cylindrical\n"
        <<"6) 3D Spherical\n";
    int mode=askIntRange("> ",0,6,0);
    if(mode==0) mode=(inferred>=0?inferred:0);

    if(mode==1){
        int k=(inferred==1&& !loaded.empty())?(int)loaded.front().size():askIntRange("How many variables per row? ",2,64,3);
        if(k<2||k>64){cout<<"Invalid variable count.\n";return;}
        vector<vector<ld>>rows;for(auto&r:loaded){if((int)r.size()==k)rows.push_back(r);}if(rows.size()<8){cout<<"Need at least 8 complete rows with exactly "<<k<<" variables.\n";return;}
        ForecastOptions o=askForecastOptions(false);o.requestPlot=askYesNo("Generate a plot of the multivariate driver/forecast?",false);if(g_inputClosed)return;
        MultiAnalysis ma=analyzeMultivariate(rows,o.horizon,o.customTolerance?o.tolerancePct:-1.0L);printMultivariate(ma,o.horizon);
        if(!ma.fullForecastAvailable){vector<ld>v;for(auto&r:rows)v.push_back(r[0]);printCompactFallbackDiagnostics(v,ma.failureReason,o.customTolerance?o.tolerancePct:-1.0L);}else if(o.requestPlot){vector<Point>q;for(size_t i=0;i<rows.size();++i)q.push_back({(ld)i+1,rows[i][0],0});writeMultivariatePlotFiles(rows,ma.predictions);writePlotFiles(q,ma.driver.model.valid?&ma.driver.model:nullptr,"Multivariate driver / chained forecast",false,o.horizon,1.0L);asciiPlot(q,ma.driver.model.valid?&ma.driver.model:nullptr);cout<<"Plot output: multivariate_analysis.png and sequence_analysis.png.\n";}
        return;
    }

    vector<Point>pts;vector<ld>nums;vector<ld>x;bool rad=true;bool is3D=(mode>=4);
    if(mode==0){for(auto&r:loaded)for(ld v:r){pts.push_back({(ld)pts.size()+1,v,0});nums.push_back(v);}}
    else if(mode==2){for(auto&r:loaded)if(r.size()>=2)pts.push_back({r[0],r[1],0});for(auto&p:pts)nums.push_back(p.y);}
    else {
        if(mode==3||mode==5||mode==6){int u=askIntRange("Angle units: 1) radians  2) degrees\n> ",1,2,1);rad=(u==1);}
        for(auto&r:loaded){if(mode==3&&r.size()>=2){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),0});nums.push_back(pts.back().y);}else if(mode==4&&r.size()>=3){pts.push_back({r[0],r[1],r[2]});nums.push_back(r[2]);}else if(mode==5&&r.size()>=3){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),r[2]});nums.push_back(r[2]);}else if(mode==6&&r.size()>=3){ld th=rad?r[1]:r[1]*PI/180,ph=rad?r[2]:r[2]*PI/180;pts.push_back({r[0]*sinl(ph)*cosl(th),r[0]*sinl(ph)*sinl(th),r[0]*cosl(ph)});nums.push_back(pts.back().z);}}
    }
    if(nums.size()<3){cout<<"Need at least 3 usable observations.\n";return;}
    ForecastOptions o=askForecastOptions(true);x.resize(nums.size());iota(x.begin(),x.end(),1.0L);if(mode==2){x.clear();for(auto&p:pts)x.push_back(p.x);sort(pts.begin(),pts.end(),[](const Point&a,const Point&b){return a.x<b.x;});x.clear();nums.clear();for(auto&p:pts){x.push_back(p.x);nums.push_back(p.y);}}
    Analysis res=analyzeSequence(x,nums,o.horizon,o.customTolerance?o.tolerancePct:-1.0L);printAnalysis(res,o.horizon);
    if(!res.found)printCompactFallbackDiagnostics(nums,res.warning,o.customTolerance?o.tolerancePct:-1.0L);else if(o.requestPlot)writePlotFiles(pts,res.model.valid?&res.model:nullptr,res.model.valid?res.model.formula:"No validated model",is3D,o.horizon,mode==2?(x.size()>1?x.back()-x[x.size()-2]:1.0L):1.0L);
}


string arrayHex(const array<uint8_t,32>&a){return bytesToHex(vector<uint8_t>(a.begin(),a.end()));}
bool runSelfTestCase(const string&name,bool ok){cout<<"  ["<<(ok?"PASS":"FAIL")<<"] "<<name<<"\n";return ok;}
void runSelfTestSuite(){
    printMethodCard(9);
    int pass=0,total=0;
    auto check=[&](const string&n,bool ok){++total;if(ok)++pass;return runSelfTestCase(n,ok);};
    check("SHA-256 abc", sha256Hex("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    check("HMAC-SHA256 reference", hmacSha256Hex("key","The quick brown fox jumps over the lazy dog")=="f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8");
    vector<uint8_t>salt={'s','a','l','t'};auto dk=pbkdf2HmacSha256("password",salt,1,32);
    check("PBKDF2-HMAC-SHA256 c=1", bytesToHex(dk)=="120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b");
    check("Morse encode", morseEncode("SOS") == "... --- ...");
    check("Morse decode", morseDecode("... --- ...") == "SOS");
    check("Atbash", atbash("ABC xyz") == "ZYX cba");
    check("Affine round-trip", affineTransform(affineTransform("Hello World",5,8,false),5,8,true)=="Hello World");
    vector<ld>x(10);iota(x.begin(),x.end(),1);vector<ld>y;for(ld q:x)y.push_back(2+3*q);
    auto a=analyzeSequence(x,y,3,-1);
    check("Linear extrapolation", a.found && a.predictions.size()==3 && approxEqual(a.predictions[0],35) && approxEqual(a.predictions[1],38));
    vector<ld>g={1,2,4,8,16,32,64,128};vector<ld>gx(g.size());iota(gx.begin(),gx.end(),1);auto ag=analyzeSequence(gx,g,2,-1);
    check("Exact geometric detection", ag.found && ag.deterministicNext && approxEqual(ag.predictions[0],256) && approxEqual(ag.predictions[1],512));
    vector<ld>cp={1,1,1,1,1,20,20,20,20,20,20,20};
    auto cps=detectChangePoints(cp);
    check("Change-point diagnostics runs", !cps.empty());
    {
        vector<ld> sx; vector<ld> sy;
        for(int i=0;i<42;++i){ld z=-2.5L+0.113636363636L*i; sx.push_back(z);ld q=sinl(expl(-z*z));sy.push_back(tanhl(cosl(logl(q))));}
        auto as=analyzeSequence(sx,sy,4,-1);
        ld truth=tanhl(cosl(logl(sinl(expl(-sx.back()*sx.back())))));
        (void)truth;
        ld z=sx.back()+0.113636363636L, expected=tanhl(cosl(logl(sinl(expl(-z*z)))));
        check("Nested analytic transform recognition", as.found && as.model.predict && fabsl(as.model.predict(z)-expected)<0.01L);
        check("Numerical root localization", !numericRealRoots([&](ld q){return tanhl(cosl(logl(sinl(expl(-q*q)))));},-2.5L,2.5L,2048).empty());
    }
    {
        vector<ld> sx,sy;for(int i=-5;i<=5;++i){sx.push_back(i);sy.push_back((ld)i*i-1.0L);}
        auto rg=fitRootGuidedPolynomial(sx,sy,2,2);
        bool rootsOK=rg.valid&&rg.approxRealRoots.size()==2&&fabsl(rg.approxRealRoots[0]+1.0L)<1e-4L&&fabsl(rg.approxRealRoots[1]-1.0L)<1e-4L;
        check("Root-guided polynomial fit",rootsOK);
    }
    {
        vector<ld> zx(14),zy; iota(zx.begin(),zx.end(),1); for(ld q:zx)zy.push_back(powl(2,q)+powl(3,q));
        auto pr=fitPronyZModel(zx,zy,2);
        check("Prony / Z-transform recurrence",pr.valid&&pr.approxRealRoots.size()>=2&&fabsl(pr.approxRealRoots[0]-2)<1e-5L&&fabsl(pr.approxRealRoots[1]-3)<1e-5L);
    }
    {
        vector<ld> sx(20),sy={10,10.1,9.9,10.05,10.02,10.1,10.0,19.9,20.1,20.0,19.8,20.2,20.0,20.1,29.9,30.1,30.0,29.8,30.2,30.0};
        iota(sx.begin(),sx.end(),1); auto pc=fitPiecewiseConstantModel(sx,sy);
        check("Noisy zero-order-hold step fit",pc.valid&&pc.predict&&fabsl(pc.predict(21)-30.0L)<1e-6L);
    }
    {
        vector<ld> sx(12);iota(sx.begin(),sx.end(),1);vector<ld> sy={2,2,2,2,5,5,5,5,9,9,9,9};
        auto st=analyzeSequence(sx,sy,2,-1);
        check("Exact zero-order-hold step forecast", st.found && st.deterministicNext && approxEqual(st.predictions[0],9) && approxEqual(st.predictions[1],9));
    }
    {
        vector<ld> sx(20),sy; iota(sx.begin(),sx.end(),1);
        for(int i=0;i<20;++i){int block=i/2+1;sy.push_back((ld)block);}
        auto ast=analyzeSequence(sx,sy,4,-1);
        check("Repeated staircase continues levels",ast.found&&ast.predictions.size()>=4&&approxEqual(ast.predictions[0],11)&&approxEqual(ast.predictions[1],11)&&approxEqual(ast.predictions[2],12)&&approxEqual(ast.predictions[3],12));
    }
    {
        vector<ld> sx(24),sy; iota(sx.begin(),sx.end(),1);
        const ld phase[4]={0,10,-3,7};
        for(int i=1;i<=24;++i) sy.push_back(100+0.5L*i+phase[(i%4+4)%4]);
        auto sm=fitSeasonalTemplateModel(sx,sy,4,1);
        check("Seasonal template detects period",sm.valid&&sm.predict&&fabsl(sm.predict(25)-(100+0.5L*25+phase[1]))<1e-6L);
    }
    {
        vector<ld> sx(24),sy; iota(sx.begin(),sx.end(),1);
        const ld delta[4]={1,-2,3,0};
        sy={10,20,30,40};
        for(int i=4;i<24;++i) sy.push_back(sy[i-4]+delta[((i+1)%4+4)%4]);
        auto dm=fitPeriodicDifferenceModel(sx,sy,4);
        ld e1=sy[21];
        ld expected=sy[20]+delta[(25%4+4)%4];
        check("Periodic-difference recurrence",dm.valid&&dm.predict&&fabsl(dm.predict(25)-expected)<1e-8L&&fabsl(dm.predict(25)-e1)>0.1L);
    }
    {
        vector<ld> sx(20),sy; iota(sx.begin(),sx.end(),1);
        for(int i=1;i<=20;++i) sy.push_back(i<=10 ? 2.0L*i+1.0L : 100.0L+(i-10)*(i-10));
        auto pm=fitPiecewiseAdaptiveModel(sx,sy,2);
        check("Adaptive piecewise segment model",pm.valid&&pm.predict&&fabsl(pm.predict(21)-221.0L)<1e-4L);
    }
    {
        vector<ld> sx;for(int i=0;i<=10;++i)sx.push_back(i); vector<ld> sy;for(ld z:sx)sy.push_back((z-100)*(z-100));
        auto rm=fitPolyModel(sx,sy,2); attachNumericRoots(rm,sx,true); bool ok=false; for(ld r:rm.approxRealRoots) if(fabsl(r-100.0L)<1e-4L) ok=true;
        check("Numerical roots extend beyond observed interval",rm.valid&&ok);
    }
    {
        auto a=parseNumberLine(" (1, 393) ; x=5, y=7 ");
        check("Flexible numeric input syntax",a.size()==4&&approxEqual(a[0],1)&&approxEqual(a[1],393)&&approxEqual(a[2],5)&&approxEqual(a[3],7));
    }
    {
        vector<ld> px,py; for(int i=1;i<=30;++i){px.push_back(i);py.push_back(i<=15?2.0L*i+1.0L:20.0L+5.0L*expl(0.15L*(i-15)));}
        auto hp=fitPiecewiseAdaptiveModel(px,py,2);
        check("Piecewise mixed linear/exponential domains",hp.valid&&hp.predict&&fabsl(hp.predict(31)-(20.0L+5.0L*expl(2.4L)))<1.0L);
    }
    {
        vector<ld> z(12),q; iota(z.begin(),z.end(),0.0L); for(int i=0;i<12;++i)q.push_back(powl(2.0L,i)+powl(3.0L,i));
        auto pr=fitPronyZModel(z,q,2); auto pp=forecastFromModel(pr,z.back(),2); check("Zero-based Prony forecast indexing",pr.valid&&pp.size()==2&&fabsl(pp[0]-535537.0L)<1e-5L&&fabsl(pp[1]-1602515.0L)<1e-4L);
    }
    {
        vector<ld> sx(48),sy; iota(sx.begin(),sx.end(),0.0L);
        for(int i=0;i<48;++i){int k=i/3; sy.push_back(5.0L+1.75L*k+((i%3)-1)*0.06L);}
        auto rs=fitRobustStaircaseProgression(sx,sy);
        check("Noise-tolerant staircase event progression",rs.valid&&rs.predict&&fabsl(rs.predict(48)-33.0L)<0.5L);
    }
    {
        vector<ld> hx(31),hy={3.000000L,4.381773L,4.493151L,3.151127L,2.589553L,4.324738L,10.680754L,20.410888L,29.843858L,32.455979L,27.191070L,19.999990L,18.463427L,25.420167L,43.990607L,70.650288L,85.712096L,75.038603L,38.249013L,3.149877L,-12.087055L,8.997799L,68.031959L,142.425887L,162.094421L,106.867649L,22.192620L,-11.232533L,41.260501L,146.792229L,188.600130L};
        iota(hx.begin(),hx.end(),0.0L);
        auto turns=observedTurningPoints(hx,hy);auto infl=observedInflectionPoints(hx,hy);auto zr=observedZeroCrossings(hx,hy);ld ep=estimateExtremaPeriod(hx,hy);auto om=fitOscillatoryEnvelopeModel(hx,hy,ep);
        check("Hard oscillatory curve has extrema/period diagnostics",turns.size()>=5&&infl.size()>=3&&zr.size()>=2&&isfinite(ep));
        check("Turning-point guided oscillatory fallback fits hard curve",om.valid&&om.predict&&isfinite(om.predict(31)));
        auto ha=analyzeSequence(hx,hy,4,-1);
        check("Hard oscillatory curve produces shape-aware forecast",ha.found&&ha.shapeBased&&ha.predictions.size()==4);
        if(om.valid){attachNumericRoots(om,hx,true);check("Hard curve numerical roots + extrema diagnostics",!om.approxDerivativeRoots.empty()&&!om.approxInflectionRoots.empty());}
    }
    {
        vector<ld> rx;vector<ld> ry;for(int i=0;i<=30;++i){ld z=i;rx.push_back(z);ry.push_back((z<10)?2*z+1:20+4*expl(0.12L*(z-10)));}
        auto pa=fitPiecewiseAdaptiveModel(rx,ry,2);
        check("Piecewise discontinuity / mixed nonlinear domains",pa.valid&&pa.predict&&isfinite(pa.predict(31)));
    }
    {
        vector<ld> rx;vector<ld> ry;for(int i=0;i<=40;++i){ld z=i-20.0L;rx.push_back(z);ry.push_back((z+2.0L)*z*(z-2.0L));}
        auto pm=fitPolyModel(rx,ry,3);attachNumericRoots(pm,rx,true);bool r1=false,r2=false,r3=false;for(ld q:pm.approxRealRoots){r1|=fabsl(q+2)<1e-5L;r2|=fabsl(q-0)<1e-5L;r3|=fabsl(q-2)<1e-5L;}
        check("Three-root numerical polynomial localization",r1&&r2&&r3);
        check("Second-derivative inflection localization",!pm.approxInflectionRoots.empty());
    }
    {
        auto cm=makeCubeMoves(2);vector<int>state(24);for(int f=0;f<6;++f)for(int i=0;i<4;++i)state[f*4+i]=f;vector<int>q=state;int u=cubeMoveIndex(cm,"U");for(int i=0;i<4;++i)q=applyMove(q,cm[u]);check("2x2 cube face-turn identity",q==state);
        IDAStarSolver ps;ps.moves={{"swap",{1,0},0}};auto sol=ps.solve({1,0},{0,1},1,1000);check("Generic permutation puzzle solver",sol.size()==1 && sol[0]==0);
    }
#ifdef ENGINE_USE_OPENSSL
    { string keyHex(64,'0'),ivHex(24,'0'),plain="OpenSSL-GCM-test"; auto enc=aesGcmEncryptOpenSSL(keyHex,plain,ivHex); bool ok=false; if(!enc.empty()){auto pos=enc.find(':'); if(pos!=string::npos){auto dec=aesGcmDecryptOpenSSL(keyHex,enc.substr(0,pos),ivHex,enc.substr(pos+1)); ok=(dec==plain);}} check("AES-256-GCM round-trip",ok); }
#endif
    cout<<"\nSelf-test result: "<<pass<<" / "<<total<<" checks passed.\n";
    cout<<"Note: model-selection behavior can vary with platform floating-point details; cryptographic known-answer tests should be exact.\n";
}

// ----------------------------- Web / API bridge ------------------
// The web bridge is additive. It does not replace the native console or any
// of the analysis algorithms above. Each web command calls the same C++
// functions used by the interactive application and serializes their result
// for FastAPI/React.
string webReadAll(){
    ostringstream ss;
    ss << cin.rdbuf();
    return ss.str();
}

vector<ld> webReadNumbers(const string& text){
    vector<ld> out;
    string line;
    istringstream in(text);
    bool sawData=false;
    while(getline(in,line)){
        if(line.find_first_not_of(" \t\r\n")==string::npos) continue;
        auto values=parseNumberLine(line);
        if(values.empty()){
            if(!sawData) continue; // friendly CSV/TSV header support
            return {};
        }
        sawData=true;
        out.insert(out.end(),values.begin(),values.end());
    }
    return out;
}

bool webReadXYPairs(const string& text,vector<ld>&x,vector<ld>&y){
    x.clear(); y.clear();
    string line; istringstream in(text);
    bool sawData=false;
    while(getline(in,line)){
        if(line.find_first_not_of(" \t\r\n")==string::npos) continue;
        auto v=parseNumberLine(line);
        if(v.size()!=2){
            if(!sawData) continue; // tolerate a first header row
            return false;
        }
        sawData=true;
        x.push_back(v[0]); y.push_back(v[1]);
    }
    return x.size()>=3;
}

string webJsonEscape(const string&s){
    string out; out.reserve(s.size()+8); out.push_back('"');
    for(unsigned char c: s){
        switch(c){
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if(c<0x20){
                    char buf[7]; snprintf(buf,sizeof(buf),"\\u%04x",(unsigned)c); out += buf;
                }else out.push_back((char)c);
        }
    }
    out.push_back('"'); return out;
}

string webJsonNumber(ld v){
    if(!isfinite(v)) return "null";
    ostringstream ss; ss << setprecision(numeric_limits<ld>::max_digits10) << v; return ss.str();
}

string webJsonInt(int v){ return to_string(v); }

string webJsonBool(bool v){ return v ? "true" : "false"; }

string webJsonArray(const vector<ld>&v){
    ostringstream ss; ss << '[';
    for(size_t i=0;i<v.size();++i){ if(i) ss << ','; ss << webJsonNumber(v[i]); }
    ss << ']'; return ss.str();
}

string webJsonIntArray(const vector<int>&v){
    ostringstream ss; ss << '[';
    for(size_t i=0;i<v.size();++i){ if(i) ss << ','; ss << v[i]; }
    ss << ']'; return ss.str();
}

string webJsonStringArray(const vector<string>&v){
    ostringstream ss; ss << '[';
    for(size_t i=0;i<v.size();++i){ if(i) ss << ','; ss << webJsonEscape(v[i]); }
    ss << ']'; return ss.str();
}

string webMetricsJson(const Metrics&m){
    ostringstream ss;
    ss << '{'
       << "\"nTrain\":" << m.nTrain << ','
       << "\"nValid\":" << m.nValid << ','
       << "\"validationBlocks\":" << m.validationBlocks << ','
       << "\"rmse\":" << webJsonNumber(m.rmse) << ','
       << "\"mae\":" << webJsonNumber(m.mae) << ','
       << "\"r2\":" << webJsonNumber(m.r2) << ','
       << "\"normalizedRMSE\":" << webJsonNumber(m.normalizedRMSE) << ','
       << "\"hitRate\":" << webJsonNumber(m.hitRate) << ','
       << "\"meanDeviationPct\":" << webJsonNumber(m.meanDeviationPct) << ','
       << "\"maxDeviationPct\":" << webJsonNumber(m.maxDeviationPct) << ','
       << "\"p90DeviationPct\":" << webJsonNumber(m.p90DeviationPct) << ','
       << "\"smapePct\":" << webJsonNumber(m.smapePct) << ','
       << "\"directionalAccuracy\":" << webJsonNumber(m.directionalAccuracy) << ','
       << "\"correlation\":" << webJsonNumber(m.correlation) << ','
       << "\"wapePct\":" << webJsonNumber(m.wapePct) << ','
       << "\"mase\":" << webJsonNumber(m.mase) << ','
       << "\"medianDeviationPct\":" << webJsonNumber(m.medianDeviationPct) << ','
       << "\"blockPassRate\":" << webJsonNumber(m.blockPassRate) << ','
       << "\"score\":" << webJsonNumber(m.score) << ','
       << "\"stable\":" << webJsonBool(m.stable) << ','
       << "\"rollingValidated\":" << webJsonBool(m.rollingValidated) << ','
       << "\"scoreRMSE\":" << webJsonNumber(m.scoreRMSE) << ','
       << "\"scoreMAE\":" << webJsonNumber(m.scoreMAE) << ','
       << "\"scoreP90\":" << webJsonNumber(m.scoreP90) << ','
       << "\"scoreR2\":" << webJsonNumber(m.scoreR2) << ','
       << "\"scoreDirection\":" << webJsonNumber(m.scoreDirection) << ','
       << "\"scoreCorrelation\":" << webJsonNumber(m.scoreCorrelation) << ','
       << "\"scoreStability\":" << webJsonNumber(m.scoreStability) << ','
       << "\"scoreComplexity\":" << webJsonNumber(m.scoreComplexity) << ','
       << "\"blockMeanDeviationPct\":" << webJsonArray(m.blockMeanDeviationPct) << ','
       << "\"blockMaxDeviationPct\":" << webJsonArray(m.blockMaxDeviationPct)
       << '}';
    return ss.str();
}

string webStatsJson(const Stats&s){
    ostringstream ss;
    ss << '{'
       << "\"n\":" << s.n << ','
       << "\"mean\":" << webJsonNumber(s.mean) << ','
       << "\"median\":" << webJsonNumber(s.median) << ','
       << "\"variance\":" << webJsonNumber(s.variance) << ','
       << "\"sd\":" << webJsonNumber(s.sd) << ','
       << "\"min\":" << webJsonNumber(s.minv) << ','
       << "\"max\":" << webJsonNumber(s.maxv) << ','
       << "\"q1\":" << webJsonNumber(s.q1) << ','
       << "\"q3\":" << webJsonNumber(s.q3) << ','
       << "\"iqr\":" << webJsonNumber(s.iqr) << ','
       << "\"mad\":" << webJsonNumber(s.mad) << ','
       << "\"skew\":" << webJsonNumber(s.skew) << ','
       << "\"excessKurtosis\":" << webJsonNumber(s.excessKurtosis) << ','
       << "\"lag1\":" << webJsonNumber(s.lag1) << ','
       << "\"durbinWatson\":" << webJsonNumber(s.durbinWatson) << ','
       << "\"runsZ\":" << webJsonNumber(s.runsZ) << ','
       << "\"jarqueBera\":" << webJsonNumber(s.jarqueBera) << ','
       << "\"jarqueBeraP\":" << webJsonNumber(s.jarqueBeraP) << ','
       << "\"standardError\":" << webJsonNumber(s.standardError) << ','
       << "\"ci95Low\":" << webJsonNumber(s.ci95Low) << ','
       << "\"ci95High\":" << webJsonNumber(s.ci95High)
       << '}';
    return ss.str();
}

string webModelJson(const Model&m,bool includeFitted=true){
    ostringstream ss;
    ss << '{'
       << "\"name\":" << webJsonEscape(m.name) << ','
       << "\"category\":" << webJsonEscape(m.category) << ','
       << "\"formula\":" << webJsonEscape(m.formula) << ','
       << "\"complexity\":" << m.complexity << ','
       << "\"valid\":" << webJsonBool(m.valid) << ','
       << "\"exact\":" << webJsonBool(m.exact) << ','
       << "\"nextValue\":" << webJsonNumber(m.nextValue) << ','
       << "\"window\":" << m.window << ','
       << "\"note\":" << webJsonEscape(m.note) << ','
       << "\"approxRealRoots\":" << webJsonArray(m.approxRealRoots) << ','
       << "\"approxDerivativeRoots\":" << webJsonArray(m.approxDerivativeRoots) << ','
       << "\"approxInflectionRoots\":" << webJsonArray(m.approxInflectionRoots) << ','
       << "\"calculusSummary\":" << webJsonEscape(m.calculusSummary) << ','
       << "\"metrics\":" << webMetricsJson(m.metrics);
    if(includeFitted) ss << ",\"fitted\":" << webJsonArray(m.fitted);
    ss << '}'; return ss.str();
}

string webAnalysisJson(const Analysis&a,const vector<ld>*xOverride=nullptr,const vector<ld>*yOverride=nullptr,ld trajectoryStepOverride=numeric_limits<ld>::quiet_NaN()){
    ostringstream ss;
    ss << '{'
       << "\"found\":" << webJsonBool(a.found) << ','
       << "\"deterministicNext\":" << webJsonBool(a.deterministicNext) << ','
       << "\"category\":" << webJsonEscape(a.category) << ','
       << "\"logic\":" << webJsonEscape(a.logic) << ','
       << "\"formula\":" << webJsonEscape(a.formula) << ','
       << "\"nextValue\":" << webJsonNumber(a.nextValue) << ','
       << "\"evidenceScore\":" << webJsonNumber(a.evidenceScore) << ','
       << "\"tolerancePct\":" << webJsonNumber(a.tolerancePct) << ','
       << "\"warning\":" << webJsonEscape(a.warning) << ','
       << "\"model\":" << webModelJson(a.model,true) << ','
       << "\"predictions\":" << webJsonArray(a.predictions) << ','
       << "\"predictionLow\":" << webJsonArray(a.predictionLow) << ','
       << "\"predictionHigh\":" << webJsonArray(a.predictionHigh) << ','
       << "\"changePoints\":" << webJsonIntArray(a.changePoints) << ','
       << "\"observedZeroCrossings\":" << webJsonArray(a.observedZeroCrossings) << ','
       << "\"observedTurningPoints\":" << webJsonArray(a.observedTurningPoints) << ','
       << "\"observedInflectionPoints\":" << webJsonArray(a.observedInflectionPoints) << ','
       << "\"calculusSummary\":" << webJsonEscape(a.calculusSummary) << ','
       << "\"shapeBased\":" << webJsonBool(a.shapeBased) << ',';
    if(xOverride){
        ss << "\"x\":" << webJsonArray(*xOverride) << ',';
        vector<ld>tx=*xOverride,ty=yOverride?*yOverride:vector<ld>{};
        if(yOverride && !a.predictions.empty()){
            ld step=isfinite(trajectoryStepOverride)&&trajectoryStepOverride>0?trajectoryStepOverride:1.0L;
            if(!(isfinite(trajectoryStepOverride)&&trajectoryStepOverride>0) && xOverride->size()>=2) step=(*xOverride)[xOverride->size()-1]-(*xOverride)[xOverride->size()-2];
            if(fabsl(step)<1e-15L)step=1.0L;
            for(size_t h=0;h<a.predictions.size();++h){tx.push_back((*xOverride).back()+step*(ld)(h+1));ty.push_back(a.predictions[h]);}
            ss << "\"trajectoryX\":" << webJsonArray(tx) << ',';
            ss << "\"trajectoryY\":" << webJsonArray(ty) << ',';
        }
    }
    if(yOverride) ss << "\"y\":" << webJsonArray(*yOverride) << ',';
    ss << "\"leaderboard\":[";
    for(size_t i=0;i<a.leaderboard.size();++i){
        if(i) ss << ',';
        ss << webModelJson(a.leaderboard[i],false);
    }
    ss << "]}";
    return ss.str();
}

string webOk(const string&body){ return string("{\"ok\":true,") + body + "}"; }
string webFail(const string&message){ return string("{\"ok\":false,\"error\":") + webJsonEscape(message) + "}"; }

bool webParseInt(const string&s,int&out){
    try{ size_t p=0; long long q=stoll(s,&p); if(p!=s.size()||q<INT_MIN||q>INT_MAX)return false; out=(int)q; return true; }
    catch(...){ return false; }
}

bool webParseLD(const string&s,ld&out){
    try{ size_t p=0; ld q=stold(s,&p); if(p!=s.size()||!isfinite(q))return false; out=q; return true; }
    catch(...){ return false; }
}

vector<ld> webDenseGrid(ld xmin,ld xmax,int count){
    count=max(40,min(count,1000));
    vector<ld>g(count);
    for(int i=0;i<count;++i)g[i]=xmin+(xmax-xmin)*(ld)i/max(1,count-1);
    return g;
}

vector<ld> webModelCurve(const Model&m,const vector<ld>&grid){
    vector<ld>out; out.reserve(grid.size());
    if(!m.predict) return out;
    for(ld z:grid){ ld v=m.predict(z); out.push_back(isfinite(v)?v:numeric_limits<ld>::quiet_NaN()); }
    return out;
}

struct WebFamilyCandidate { string family; FamilySpec spec; Model model; };

vector<WebFamilyCandidate> webFitFamilyCandidates(const vector<ld>&x,const vector<ld>&y){
    vector<WebFamilyCandidate>out;
    auto specs=makeSpecs(x,y);
    out.reserve(specs.size());
    for(const auto&spec:specs){
        Model m=fitByFamily(spec.family,x,y,spec.degree,spec.param,spec.param2);
        if(!m.valid || !m.predict) continue;
        m.fitted=finiteFitted(m,x,y);
        auto [rmse,mae]=errorMetrics(y,m.fitted);
        m.metrics.rmse=rmse; m.metrics.mae=mae; m.metrics.r2=r2Score(y,m.fitted);
        m.nextValue=m.predict(x.back());
        attachNumericRoots(m,x,true);
        out.push_back({spec.family,spec,m});
    }
    stable_sort(out.begin(),out.end(),[](const WebFamilyCandidate&a,const WebFamilyCandidate&b){
        if(a.model.metrics.rmse!=b.model.metrics.rmse)return a.model.metrics.rmse<b.model.metrics.rmse;
        if(a.model.metrics.mae!=b.model.metrics.mae)return a.model.metrics.mae<b.model.metrics.mae;
        if(a.model.complexity!=b.model.complexity)return a.model.complexity<b.model.complexity;
        return a.model.name<b.model.name;
    });
    return out;
}

string webFamilyCandidateJson(const WebFamilyCandidate&c){
    ostringstream ss;
    ss << '{'
       << "\"family\":" << webJsonEscape(c.family) << ','
       << "\"degree\":" << c.spec.degree << ','
       << "\"param\":" << webJsonNumber(c.spec.param) << ','
       << "\"param2\":" << webJsonNumber(c.spec.param2) << ','
       << "\"model\":" << webModelJson(c.model,true)
       << '}';
    return ss.str();
}

string webStatsFromNumbers(const vector<ld>&nums){
    if(nums.empty()) return webFail("No valid numeric observations supplied.");
    return webOk(string("\"stats\":") + webStatsJson(computeStats(nums)) + ",\"values\":" + webJsonArray(nums));
}

string webSequenceCommand(int horizon,ld tol){
    const string text=webReadAll();
    auto y=webReadNumbers(text);
    if(y.size()<3)return webFail("Dataset must contain at least 3 numerical values.");
    vector<ld>x(y.size()); iota(x.begin(),x.end(),1.0L);
    Analysis a=analyzeSequence(x,y,horizon,tol);
    return webOk(string("\"analysis\":")+webAnalysisJson(a,&x,&y)+",\"stats\":"+webStatsJson(computeStats(y)));
}

string webRunCurve(int horizon,ld tol,ld step,bool includeFamilies){
    vector<ld>x,y;
    if(!webReadXYPairs(webReadAll(),x,y)) return webFail("Enter at least 3 valid x y pairs, one pair per line.");
    vector<pair<ld,ld>> pairs; for(size_t i=0;i<x.size();++i)pairs.push_back({x[i],y[i]});
    sort(pairs.begin(),pairs.end()); x.clear();y.clear(); for(auto&p:pairs){x.push_back(p.first);y.push_back(p.second);}
    for(size_t i=1;i<x.size();++i) if(fabsl(x[i]-x[i-1])<1e-15L) return webFail("Duplicate x values are not supported in curve regression mode.");

    Analysis a=analyzeSequence(x,y,horizon,tol);
    vector<ld>futureX; futureX.reserve(horizon);
    for(int h=1;h<=horizon;++h)futureX.push_back(x.back()+step*h);
    if(a.model.valid && a.model.predict){
        a.predictions.clear();
        for(ld fx:futureX){ld p=a.model.predict(fx);if(!isfinite(p)){a.predictions.clear();break;}a.predictions.push_back(p);}
        a.nextValue=a.predictions.empty()?numeric_limits<ld>::quiet_NaN():a.predictions.front();
        a.model.nextValue=a.nextValue;
        auto half=empiricalForecastBand(a.predictions,a.model.validationResiduals,0.90L);
        if(half.size()==a.predictions.size()){
            a.predictionLow.resize(half.size());a.predictionHigh.resize(half.size());
            for(size_t i=0;i<half.size();++i){a.predictionLow[i]=a.predictions[i]-half[i];a.predictionHigh[i]=a.predictions[i]+half[i];}
        }
    }
    ld xmin=x.front(),xmax=x.back();
    if(!futureX.empty())xmax=max(xmax,futureX.back());
    vector<ld>grid=webDenseGrid(xmin,xmax,500);
    vector<ld>selectedCurve=webModelCurve(a.model,grid);

    ostringstream body;
    body << "\"analysis\":" << webAnalysisJson(a,&x,&y,step)
         << ",\"stats\":" << webStatsJson(computeStats(y))
         << ",\"futureX\":" << webJsonArray(futureX)
         << ",\"gridX\":" << webJsonArray(grid)
         << ",\"gridY\":" << webJsonArray(selectedCurve)
         << ",\"step\":" << webJsonNumber(step);

    if(includeFamilies){
        auto candidates=webFitFamilyCandidates(x,y);
        body << ",\"families\":[";
        for(size_t i=0;i<candidates.size();++i){if(i)body<<',';body<<webFamilyCandidateJson(candidates[i]);}
        body << ']';
        body << ",\"familyGridX\":" << webJsonArray(grid);
        body << ",\"familyCurves\":[";
        const size_t limit=min<size_t>(12,candidates.size());
        for(size_t i=0;i<limit;++i){
            if(i)body<<',';
            body << '{' << "\"family\":" << webJsonEscape(candidates[i].family)
                 << ",\"name\":" << webJsonEscape(candidates[i].model.name)
                 << ",\"curve\":" << webJsonArray(webModelCurve(candidates[i].model,grid)) << '}';
        }
        body << ']';
    }
    return webOk(body.str());
}


string webFamilySweepCommand(){
    vector<ld>x,y;
    if(!webReadXYPairs(webReadAll(),x,y))return webFail("Enter at least 3 valid x y pairs, one pair per line.");
    vector<pair<ld,ld>>pairs;for(size_t i=0;i<x.size();++i)pairs.push_back({x[i],y[i]});sort(pairs.begin(),pairs.end());x.clear();y.clear();for(auto&p:pairs){x.push_back(p.first);y.push_back(p.second);} 
    for(size_t i=1;i<x.size();++i)if(fabsl(x[i]-x[i-1])<1e-15L)return webFail("Duplicate x values are not supported.");
    auto candidates=webFitFamilyCandidates(x,y);
    vector<ld>grid=webDenseGrid(x.front(),x.back(),600);
    ostringstream body;
    body<<"\"stats\":"<<webStatsJson(computeStats(y))
        <<",\"x\":"<<webJsonArray(x)
        <<",\"y\":"<<webJsonArray(y)
        <<",\"gridX\":"<<webJsonArray(grid)
        <<",\"families\":[";
    for(size_t i=0;i<candidates.size();++i){if(i)body<<',';body<<webFamilyCandidateJson(candidates[i]);}
    body<<"],\"familyCurves\":[";
    size_t limit=min<size_t>(16,candidates.size());
    for(size_t i=0;i<limit;++i){
        if(i)body<<',';
        body<<'{'
            <<"\"family\":"<<webJsonEscape(candidates[i].family)
            <<",\"name\":"<<webJsonEscape(candidates[i].model.name)
            <<",\"curve\":"<<webJsonArray(webModelCurve(candidates[i].model,grid))
            <<'}';
    }
    body<<']';
    return webOk(body.str());
}

string webFamilyFitCommand(const string&family,int degree,ld param,ld param2,int horizon,ld step){

    vector<ld>x,y;
    if(!webReadXYPairs(webReadAll(),x,y))return webFail("Enter at least 3 valid x y pairs, one pair per line.");
    vector<pair<ld,ld>>pairs;for(size_t i=0;i<x.size();++i)pairs.push_back({x[i],y[i]});sort(pairs.begin(),pairs.end());x.clear();y.clear();for(auto&p:pairs){x.push_back(p.first);y.push_back(p.second);} 
    for(size_t i=1;i<x.size();++i)if(fabsl(x[i]-x[i-1])<1e-15L)return webFail("Duplicate x values are not supported.");
    Model m=fitByFamily(family,x,y,degree,param,param2);
    if(!m.valid||!m.predict)return webFail("The selected model family could not be fit to this dataset.");
    m.fitted=finiteFitted(m,x,y);auto [rmse,mae]=errorMetrics(y,m.fitted);m.metrics.rmse=rmse;m.metrics.mae=mae;m.metrics.r2=r2Score(y,m.fitted);attachNumericRoots(m,x,true);
    ld xmin=x.front(),xmax=x.back();vector<ld>grid=webDenseGrid(xmin,xmax,600);vector<ld>curve=webModelCurve(m,grid);
    vector<ld>futureX,futureY; for(int h=1;h<=max(1,horizon);++h){ld fx=x.back()+step*h,fv=m.predict(fx);if(!isfinite(fv)){futureX.clear();futureY.clear();break;}futureX.push_back(fx);futureY.push_back(fv);}
    return webOk(string("\"family\":")+webJsonEscape(family)+",\"x\":"+webJsonArray(x)+",\"y\":"+webJsonArray(y)+",\"model\":"+webModelJson(m,true)+",\"gridX\":"+webJsonArray(grid)+",\"gridY\":"+webJsonArray(curve)+",\"futureX\":"+webJsonArray(futureX)+",\"futureY\":"+webJsonArray(futureY));
}

string webMultivariateCommand(int variables,int horizon,ld tol){
    if(variables<2||variables>64)return webFail("Variables must be between 2 and 64.");
    const string text=webReadAll();string line;istringstream in(text);vector<vector<ld>>rows;
    while(getline(in,line)){
        if(line.find_first_not_of(" \t\r\n")==string::npos)continue;
        auto v=parseNumberLine(line);
        if(v.empty()||v.size()!=static_cast<size_t>(variables))return webFail("Every row must contain exactly "+to_string(variables)+" numeric values.");
        rows.push_back(v);
    }
    if(rows.size()<8)return webFail("Need at least 8 complete rows for multivariate validation.");
    MultiAnalysis ma=analyzeMultivariate(rows,horizon,tol);
    ostringstream ss;ss<<"\"analysis\":{";
    ss << "\"validStructure\":" << webJsonBool(ma.validStructure) << ','
       << "\"fullForecastAvailable\":" << webJsonBool(ma.fullForecastAvailable) << ','
       << "\"variables\":" << ma.variables << ','
       << "\"rows\":" << ma.rows << ','
       << "\"tolerancePct\":" << webJsonNumber(ma.tolerancePct) << ','
       << "\"failureReason\":" << webJsonEscape(ma.failureReason) << ','
       << "\"driver\":" << webAnalysisJson(ma.driver) << ','
       << "\"predictions\":[";
    for(size_t i=0;i<ma.predictions.size();++i){if(i)ss<<',';ss<<webJsonArray(ma.predictions[i]);}ss<<"],\"links\":[";
    for(size_t i=0;i<ma.links.size();++i){if(i)ss<<',';const auto&r=ma.links[i];ss<<'{'
        <<"\"found\":"<<webJsonBool(r.found)<<','
        <<"\"category\":"<<webJsonEscape(r.category)<<','
        <<"\"logic\":"<<webJsonEscape(r.logic)<<','
        <<"\"formula\":"<<webJsonEscape(r.formula)<<','
        <<"\"evidenceScore\":"<<webJsonNumber(r.evidenceScore)<<','
        <<"\"tolerancePct\":"<<webJsonNumber(r.tolerancePct)<<','
        <<"\"warning\":"<<webJsonEscape(r.warning)<<','
        <<"\"model\":"<<webModelJson(r.model,true)<<','
        <<"\"futurePredictions\":"<<webJsonArray(r.futurePredictions)<<'}';}
    ss<<"]},\"rows\":[";
    for(size_t i=0;i<rows.size();++i){if(i)ss<<',';ss<<webJsonArray(rows[i]);}ss<<"]";
    return webOk(ss.str());
}

string webCoordinatesCommand(int mode,int response,int angleUnit,int horizon,ld tol){
    if(mode<1||mode>5)return webFail("Coordinate mode must be 1-5.");
    if(response<0||response>2)return webFail("Response coordinate must be X, Y, or Z as appropriate.");
    string text=webReadAll(),line;istringstream in(text);vector<Point>pts;
    const bool is3D=mode>=3; const bool rad=angleUnit==0;
    while(getline(in,line)){
        if(line.find_first_not_of(" \t\r\n")==string::npos)continue;
        auto r=parseNumberLine(line);bool ok=false;
        if(mode==1&&r.size()==2){pts.push_back({r[0],r[1],0});ok=true;}
        else if(mode==2&&r.size()==2){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),0});ok=true;}
        else if(mode==3&&r.size()==3){pts.push_back({r[0],r[1],r[2]});ok=true;}
        else if(mode==4&&r.size()==3){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),r[2]});ok=true;}
        else if(mode==5&&r.size()==3){ld th=rad?r[1]:r[1]*PI/180,ph=rad?r[2]:r[2]*PI/180;pts.push_back({r[0]*sinl(ph)*cosl(th),r[0]*sinl(ph)*sinl(th),r[0]*cosl(ph)});ok=true;}
        if(!ok)return webFail("Malformed coordinate row for the selected coordinate system.");
    }
    if(pts.size()<3)return webFail("Need at least 3 points.");
    vector<ld>x,resp; x.reserve(pts.size());resp.reserve(pts.size());
    for(size_t i=0;i<pts.size();++i){
        if(mode==1)x.push_back(pts[i].x); else x.push_back((ld)i+1);
        resp.push_back(response==0?pts[i].x:(response==1?pts[i].y:pts[i].z));
    }
    Analysis a=analyzeSequence(x,resp,horizon,tol);
    vector<ld>grid=webDenseGrid(x.front(),x.back(),500);vector<ld>curve;
    if(a.model.predict) curve=webModelCurve(a.model,grid);
    ostringstream ss;ss<<"\"mode\":"<<mode<<",\"response\":"<<response<<",\"points\":[";
    for(size_t i=0;i<pts.size();++i){if(i)ss<<',';ss<<'{';ss<<"\"x\":"<<webJsonNumber(pts[i].x)<<",\"y\":"<<webJsonNumber(pts[i].y)<<",\"z\":"<<webJsonNumber(pts[i].z)<<'}';}
    ss<<"],\"modelX\":"<<webJsonArray(x)<<",\"responseY\":"<<webJsonArray(resp)<<",\"gridX\":"<<webJsonArray(grid)<<",\"gridY\":"<<webJsonArray(curve)<<",\"analysis\":"<<webAnalysisJson(a,&x,&resp);
    if(is3D)ss<<",\"is3D\":true";else ss<<",\"is3D\":false";
    return webOk(ss.str());
}

string webRootsCommand(){
    const string text=webReadAll(); vector<ld>x,y;
    if(!webReadXYPairs(text,x,y))return webFail("Provide at least 3 x,y rows for numerical root/calculus analysis.");
    Analysis a=analyzeSequence(x,y,1,-1);
    ostringstream ss;ss<<"\"x\":"<<webJsonArray(x)<<",\"y\":"<<webJsonArray(y)
       <<",\"observedZeroCrossings\":"<<webJsonArray(a.observedZeroCrossings)
       <<",\"observedTurningPoints\":"<<webJsonArray(a.observedTurningPoints)
       <<",\"observedInflectionPoints\":"<<webJsonArray(a.observedInflectionPoints)
       <<",\"calculusSummary\":"<<webJsonEscape(a.calculusSummary)
       <<",\"modelRoots\":"<<webJsonArray(a.model.approxRealRoots)
       <<",\"modelDerivativeRoots\":"<<webJsonArray(a.model.approxDerivativeRoots)
       <<",\"modelInflectionRoots\":"<<webJsonArray(a.model.approxInflectionRoots);
    return webOk(ss.str());
}

string webFileCommand(int mode,int variables,int angleUnit,int horizon,ld tol);

string webFilePathCommand(const string&path,int mode,int variables,int angleUnit,int horizon,ld tol){
    vector<vector<ld>>rows;string note;
    if(!loadDataRowsAuto(path,rows,note))return webFail(note);
    if(mode==0)mode=inferFileMode(rows);
    if(mode<0)return webFail("Could not infer the file structure.");
    if(mode==1){
        if(rows.empty()||(int)rows.front().size()<2||(int)rows.front().size()>64)return webFail("Invalid multivariate column count.");
        int vars=variables>1?variables:(int)rows.front().size();
        for(const auto&r:rows)if((int)r.size()!=vars)return webFail("Multivariate file has inconsistent column counts.");
        MultiAnalysis ma=analyzeMultivariate(rows,horizon,tol);
        ostringstream b;b<<"\"mode\":1,\"loadNote\":"<<webJsonEscape(note)<<",\"analysis\":{";
        b<<"\"validStructure\":"<<webJsonBool(ma.validStructure)<<",\"fullForecastAvailable\":"<<webJsonBool(ma.fullForecastAvailable)<<",\"variables\":"<<ma.variables<<",\"rows\":"<<ma.rows<<",\"tolerancePct\":"<<webJsonNumber(ma.tolerancePct)<<",\"failureReason\":"<<webJsonEscape(ma.failureReason)<<",\"driver\":"<<webAnalysisJson(ma.driver)<<",\"predictions\":[";
        for(size_t i=0;i<ma.predictions.size();++i){if(i)b<<',';b<<webJsonArray(ma.predictions[i]);}b<<"]},\"rows\":[";
        for(size_t i=0;i<rows.size();++i){if(i)b<<',';b<<webJsonArray(rows[i]);}b<<"]";return webOk(b.str());
    }
    if(mode==0){vector<ld>y;for(const auto&r:rows)for(ld q:r)y.push_back(q);if(y.size()<3)return webFail("File contains fewer than 3 numeric observations.");vector<ld>x(y.size());iota(x.begin(),x.end(),1.0L);Analysis a=analyzeSequence(x,y,horizon,tol);vector<ld>grid=webDenseGrid(x.front(),x.back(),500),curve;if(a.model.predict)curve=webModelCurve(a.model,grid);return webOk(string("\"mode\":0,\"loadNote\":")+webJsonEscape(note)+",\"x\":"+webJsonArray(x)+",\"y\":"+webJsonArray(y)+",\"gridX\":"+webJsonArray(grid)+",\"gridY\":"+webJsonArray(curve)+",\"stats\":"+webStatsJson(computeStats(y))+",\"analysis\":"+webAnalysisJson(a,&x,&y));}
    if(mode==2){vector<ld>x,y;for(const auto&r:rows)if(r.size()==2){x.push_back(r[0]);y.push_back(r[1]);}if(x.size()<3)return webFail("Need at least 3 x-y rows.");for(size_t i=1;i<x.size();++i)if(fabsl(x[i]-x[i-1])<1e-15L)return webFail("Duplicate x values.");vector<size_t>ord(x.size());iota(ord.begin(),ord.end(),0);sort(ord.begin(),ord.end(),[&](size_t a,size_t b){return x[a]<x[b];});vector<ld>sx,sy;for(auto i:ord){sx.push_back(x[i]);sy.push_back(y[i]);}Analysis a=analyzeSequence(sx,sy,horizon,tol);vector<ld>grid=webDenseGrid(sx.front(),sx.back(),500),curve;if(a.model.predict)curve=webModelCurve(a.model,grid);vector<ld>fx,fy;for(int h=1;h<=horizon;++h){ld xx=sx.back()+h, v=a.model.predict? a.model.predict(xx):numeric_limits<ld>::quiet_NaN();if(!isfinite(v))break;fx.push_back(xx);fy.push_back(v);}ostringstream b;b<<"\"mode\":2,\"loadNote\":"<<webJsonEscape(note)<<",\"x\":"<<webJsonArray(sx)<<",\"y\":"<<webJsonArray(sy)<<",\"gridX\":"<<webJsonArray(grid)<<",\"gridY\":"<<webJsonArray(curve)<<",\"futureX\":"<<webJsonArray(fx)<<",\"futureY\":"<<webJsonArray(fy)<<",\"analysis\":"<<webAnalysisJson(a,&sx,&sy);return webOk(b.str());}
    // Fall back to the existing text-based interpretation for coordinate modes.
    ostringstream raw;for(const auto&r:rows){for(size_t j=0;j<r.size();++j){if(j)raw<<' ';raw<<webJsonNumber(r[j]);}raw<<'\n';}istringstream backup(raw.str());streambuf*old=cin.rdbuf(backup.rdbuf());string result=webFileCommand(mode,variables,angleUnit,horizon,tol);cin.rdbuf(old);return result;
}

string webFileCommand(int mode,int variables,int angleUnit,int horizon,ld tol){
    const string text=webReadAll();
    if(mode==1)return webMultivariateCommand(variables,horizon,tol); // not reached with consumed stdin in normal dispatch
    string line;istringstream in(text);vector<Point>pts;vector<ld>nums;vector<ld>x;
    bool rad=angleUnit==0;bool is3D=mode>=4;
    while(getline(in,line)){
        auto r=parseNumberLine(line);if(r.empty())continue;
        if(mode==0){for(ld v:r){pts.push_back({(ld)pts.size()+1,v,0});nums.push_back(v);}}
        else if(mode==2&&r.size()==2){pts.push_back({r[0],r[1],0});nums.push_back(r[1]);}
        else if(mode==3&&r.size()==2){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),0});nums.push_back(pts.back().y);}
        else if(mode==4&&r.size()==3){pts.push_back({r[0],r[1],r[2]});nums.push_back(r[2]);}
        else if(mode==5&&r.size()==3){ld a=rad?r[1]:r[1]*PI/180;pts.push_back({r[0]*cosl(a),r[0]*sinl(a),r[2]});nums.push_back(r[2]);}
        else if(mode==6&&r.size()==3){ld th=rad?r[1]:r[1]*PI/180,ph=rad?r[2]:r[2]*PI/180;pts.push_back({r[0]*sinl(ph)*cosl(th),r[0]*sinl(ph)*sinl(th),r[0]*cosl(ph)});nums.push_back(pts.back().z);}
        else return webFail("Malformed row for selected file interpretation.");
    }
    if(nums.size()<3)return webFail("Dataset must contain at least 3 usable observations.");
    x.resize(nums.size());iota(x.begin(),x.end(),1.0L);if(mode==2){x.clear();for(auto&p:pts)x.push_back(p.x);}
    Analysis a=analyzeSequence(x,nums,horizon,tol);vector<ld>grid=webDenseGrid(x.front(),x.back(),500);vector<ld>curve;if(a.model.predict)curve=webModelCurve(a.model,grid);
    ostringstream ss;ss<<"\"mode\":"<<mode<<",\"is3D\":"<<webJsonBool(is3D)<<",\"points\":[";
    for(size_t i=0;i<pts.size();++i){if(i)ss<<',';ss<<'{';ss<<"\"x\":"<<webJsonNumber(pts[i].x)<<",\"y\":"<<webJsonNumber(pts[i].y)<<",\"z\":"<<webJsonNumber(pts[i].z)<<'}';}
    ss<<"],\"x\":"<<webJsonArray(x)<<",\"y\":"<<webJsonArray(nums)<<",\"gridX\":"<<webJsonArray(grid)<<",\"gridY\":"<<webJsonArray(curve)<<",\"stats\":"<<webStatsJson(computeStats(nums))<<",\"analysis\":"<<webAnalysisJson(a,&x,&nums);
    return webOk(ss.str());
}

string webCapture(function<void()> fn){
    ostringstream captured; streambuf* old=cout.rdbuf(captured.rdbuf());
    try{fn();}catch(...){cout.rdbuf(old);throw;} cout.rdbuf(old);return captured.str();
}

string webCipherCommand(const string&op){
    vector<string>lines;string line;istringstream in(webReadAll());while(getline(in,line))lines.push_back(line);
    auto need=[&](size_t n){return lines.size()>=n;};
    try{
        if(op=="analysis"){
            if(!need(1))return webFail("Input is required.");
            return webOk("\"text\":"+webJsonEscape(webCapture([&](){analyzeCipher(lines[0]);})));
        }
        if(op=="caesar"){
            if(!need(2)) return webFail("Provide text and shift.");
            int sh;
            if(!webParseInt(lines[1],sh)) return webFail("Invalid Caesar shift.");
            return webOk("\"result\":"+webJsonEscape(caesar(lines[0],sh)));
        }
        if(op=="vigenere"){
            if(!need(2)) return webFail("Provide ciphertext and key.");
            return webOk("\"result\":"+webJsonEscape(vigenereDecrypt(lines[0],lines[1]))+",\"ic\":"+webJsonNumber(indexOfCoincidence(lines[0])));
        }
        if(op=="xor"){
            if(!need(2)) return webFail("Provide text and key.");
            if(lines[1].empty()) return webFail("Key cannot be empty.");
            string raw=xorWithKey(lines[0],lines[1]);
            return webOk("\"hex\":"+webJsonEscape(hexEncode(raw)));
        }
        if(op=="base64_encode")return webOk("\"result\":"+webJsonEscape(need(1)?base64Encode(lines[0]):string{}));
        if(op=="base64_decode")return webOk("\"result\":"+webJsonEscape(need(1)?base64Decode(lines[0]):string{}));
        if(op=="hex_encode")return webOk("\"result\":"+webJsonEscape(need(1)?hexEncode(lines[0]):string{}));
        if(op=="hex_decode")return webOk("\"result\":"+webJsonEscape(need(1)?hexDecode(lines[0]):string{}));
        if(op=="morse_encode")return webOk("\"result\":"+webJsonEscape(need(1)?morseEncode(lines[0]):string{}));
        if(op=="morse_decode")return webOk("\"result\":"+webJsonEscape(need(1)?morseDecode(lines[0]):string{}));
        if(op=="atbash")return webOk("\"result\":"+webJsonEscape(need(1)?atbash(lines[0]):string{}));
        if(op=="affine"){
            if(!need(4)) return webFail("Provide text, a, b, decrypt(0/1).");
            int aa,bb,dec;
            if(!webParseInt(lines[1],aa)||!webParseInt(lines[2],bb)||!webParseInt(lines[3],dec)) return webFail("Invalid affine parameters.");
            return webOk("\"result\":"+webJsonEscape(affineTransform(lines[0],aa,bb,dec!=0)));
        }
        if(op=="rail_fence"){
            if(!need(3)) return webFail("Provide text, rails, mode(0 encrypt/1 decrypt).");
            int rails,dec;
            if(!webParseInt(lines[1],rails)||!webParseInt(lines[2],dec)) return webFail("Invalid Rail Fence parameters.");
            string r=dec?railFenceDecrypt(lines[0],rails):railFenceEncrypt(lines[0],rails);
            return webOk("\"result\":"+webJsonEscape(r));
        }
        if(op=="sha256")return webOk("\"result\":"+webJsonEscape(need(1)?sha256Hex(lines[0]):string{}));
        if(op=="hmac"){
            if(!need(2)) return webFail("Provide key and message.");
            return webOk("\"result\":"+webJsonEscape(hmacSha256Hex(lines[0],lines[1])));
        }
        if(op=="pbkdf2"){
            if(!need(4)) return webFail("Provide password, salt text (blank allowed), iterations, bytes.");
            int iterations,bytes;
            if(!webParseInt(lines[2],iterations)||!webParseInt(lines[3],bytes)||iterations<1||bytes<1||bytes>64) return webFail("Invalid PBKDF2 parameters.");
            vector<uint8_t>salt;
            if(lines[1].empty()) salt=secureRandomBytes(16); else salt.assign(lines[1].begin(),lines[1].end());
            auto dk=pbkdf2HmacSha256(lines[0],salt,(uint32_t)iterations,(size_t)bytes);
            return webOk("\"saltHex\":"+webJsonEscape(bytesToHex(salt))+",\"iterations\":"+to_string(iterations)+",\"keyHex\":"+webJsonEscape(bytesToHex(dk)));
        }
        if(op=="random_key"){
            if(!need(1)) return webFail("Provide byte count.");
            int n;
            if(!webParseInt(lines[0],n)||n<8||n>64) return webFail("Key bytes must be 8-64.");
            return webOk("\"hex\":"+webJsonEscape(bytesToHex(secureRandomBytes((size_t)n))));
        }
        if(op=="theory")return webOk("\"text\":"+webJsonEscape(webCapture([](){printCryptoTheory();})));
#ifdef ENGINE_USE_OPENSSL
        if(op=="aes_encrypt"){
            if(!need(3))return webFail("Provide keyHex, ivHex, plaintext.");string out=aesGcmEncryptOpenSSL(lines[0],lines[2],lines[1]);if(out.empty())return webFail("AES-256-GCM encryption failed.");auto p=out.find(':');return webOk("\"cipherHex\":"+webJsonEscape(out.substr(0,p))+",\"tagHex\":"+webJsonEscape(out.substr(p+1)));
        }
        if(op=="aes_decrypt"){
            if(!need(4))return webFail("Provide keyHex, ivHex, cipherHex, tagHex.");string out=aesGcmDecryptOpenSSL(lines[0],lines[2],lines[1],lines[3]);if(out.empty()&&!lines[2].empty())return webFail("AES-256-GCM authentication/decryption failed.");return webOk("\"plaintext\":"+webJsonEscape(out));
        }
#endif
        return webFail("Unknown crypto operation: "+op);
    }catch(const exception&e){return webFail(string("Crypto operation failed: ")+e.what());}
}

string webRandomCommand(const string&op,const vector<string>&args){
    if(op=="analyze"){
        auto nums=webReadNumbers(webReadAll());if(nums.size()<6)return webFail("Need at least 6 observations for PRNG diagnostics.");return webOk("\"stats\":"+webStatsJson(computeStats(nums))+",\"text\":"+webJsonEscape(webCapture([&](){printRandomDiagnostics(nums);}))); 
    }
    if(op=="generate"){
        if(args.size()<3) return webFail("Usage: generate <generator 1-4> <seed> <count> [m a c]");
        int gen,count;
        if(!webParseInt(args[0],gen)||!webParseInt(args[2],count)||gen<1||gen>4||count<1||count>1000000) return webFail("Invalid generator/count.");
        uint64_t seed;
        try{size_t p=0;seed=stoull(args[1],&p);if(p!=args[1].size())return webFail("Invalid seed.");}catch(...){return webFail("Invalid seed.");}
        vector<ld>out;out.reserve(count);
        if(gen==1){mt19937 rng((uint32_t)seed);uniform_int_distribution<uint32_t>d(0,numeric_limits<uint32_t>::max());for(int i=0;i<count;++i)out.push_back((ld)d(rng));}
        else if(gen==2){if(args.size()<6)return webFail("LCG generation needs modulus, multiplier and increment.");uint64_t m,a,c;try{m=stoull(args[3]);a=stoull(args[4]);c=stoull(args[5]);}catch(...){return webFail("Invalid LCG parameters.");}if(m<2||a>=m||c>=m)return webFail("Require m>=2, 0<=a<m, 0<=c<m.");uint64_t cur=seed%m;for(int i=0;i<count;++i){if(i)cur=(a*cur+c)%m;out.push_back((ld)cur);}}
        else if(gen==3){uint32_t cur=(uint32_t)seed;for(int i=0;i<count;++i){if(i)cur=xorshift32(cur);out.push_back((ld)cur);}}
        else {uint64_t cur=seed;for(int i=0;i<count;++i)out.push_back((ld)splitmix64Next(cur));}
        return webOk("\"generator\":"+to_string(gen)+",\"values\":"+webJsonArray(out));
    }
    return webFail("Unknown PRNG operation.");
}

string webSelfTestCommand(){
    return webOk("\"text\":"+webJsonEscape(webCapture([](){runSelfTestSuite();})));
}

string webPuzzleCubeCommand(int N,int depth,int nodeLimit){
    string scramble; if(!getline(cin,scramble)) return webFail("Missing scramble.");
    auto moves=makeCubeMoves(N);if(moves.empty())return webFail("Cube move construction failed.");
    vector<int>solved(6*N*N);for(int f=0;f<6;++f)for(int i=0;i<N*N;++i)solved[f*N*N+i]=f;
    vector<int>state=solved;for(const string&t:tokenizeMoves(scramble)){int mi=cubeMoveIndex(moves,t);if(mi<0)return webFail("Unknown cube move: "+t);state=applyMove(state,moves[mi]);}
    IDAStarSolver solver;solver.moves=moves;auto path=solver.solve(state,solved,depth,nodeLimit);ostringstream ss;ss<<"\"solution\":[";for(size_t i=0;i<path.size();++i){if(i)ss<<',';ss<<webJsonEscape(moves[path[i]].name);}ss<<"],\"nodes\":"<<solver.nodes<<",\"solved\":"<<webJsonBool(!path.empty()||(state==solved));return webOk(ss.str());
}

string webPuzzleGenericCommand(int n,int depth,int nodeLimit){
    string text=webReadAll();istringstream in(text);string line;if(!getline(in,line))return webFail("Missing start state.");
    auto parseState=[&](const string&s){auto v=parseNumberLine(s);if((int)v.size()!=n)return vector<int>{};vector<int>q;for(ld z:v){if(!isInteger(z)||z<-1000000||z>1000000)return vector<int>{};q.push_back((int)llroundl(z));}return q;};
    vector<int>start=parseState(line);if(start.empty())return webFail("Invalid start state.");if(!getline(in,line))return webFail("Missing goal state.");vector<int>goal=parseState(line);if(goal.empty())return webFail("Invalid goal state.");
    int mc=0;if(!getline(in,line)||!webParseInt(line,mc)||mc<1||mc>64)return webFail("Missing/invalid move count.");vector<PuzzleMove>moves;for(int i=0;i<mc;++i){if(!getline(in,line))return webFail("Missing move definition.");vector<string>parts;string p;istringstream ls(line);while(getline(ls,p,'|'))parts.push_back(p);if(parts.size()!=3)return webFail("Move format: name|group|perm");int group;if(!webParseInt(parts[1],group))return webFail("Invalid move group.");auto perm=parseNumberLine(parts[2]);if((int)perm.size()!=n)return webFail("Move permutation length mismatch.");vector<int>q;for(ld z:perm){if(!isInteger(z)||z<0||z>=n)return webFail("Move permutation contains an invalid index.");q.push_back((int)llroundl(z));}moves.push_back({parts[0],q,group});}
    IDAStarSolver solver;solver.moves=moves;auto path=solver.solve(start,goal,depth,nodeLimit);ostringstream ss;ss<<"\"solution\":[";for(size_t i=0;i<path.size();++i){if(i)ss<<',';ss<<webJsonEscape(moves[path[i]].name);}ss<<"],\"nodes\":"<<solver.nodes<<",\"solved\":"<<webJsonBool(!path.empty()||(start==goal));return webOk(ss.str());
}

// ----------------------------- Web-aware main -------------------
int main(int argc,char*argv[]){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout << std::nounitbuf;
    flushConsole();

    try{
        if(argc>=2){
            const string cmd=argv[1];
            if(cmd=="--json" && argc>=3){
                int horizon=(argc>=4)?max(1,atoi(argv[3])):1;ld tol=(argc>=5)?stold(argv[4]):-1.0L;
                cout<<webSequenceCommand(horizon,tol);return 0;
            }
            if(cmd=="--web-sequence"){
                int horizon=(argc>=3)?max(1,atoi(argv[2])):4;ld tol=(argc>=4)?stold(argv[3]):-1.0L;
                cout<<webSequenceCommand(horizon,tol);return 0;
            }
            if(cmd=="--web-auto"){
                int horizon=(argc>=3)?max(1,atoi(argv[2])):4;ld tol=(argc>=4)?stold(argv[3]):-1.0L;
                string raw=webReadAll(); vector<vector<ld>> rows; string note;
                // Reuse the same tolerant parser as the file loader.
                istringstream in(raw); string line; while(getline(in,line)){if(line.find_first_not_of(" \t\r\n")==string::npos)continue;auto v=parseNumberLine(line);if(!v.empty())rows.push_back(v);}
                int mode=inferFileMode(rows);
                if(mode==0){vector<ld>y;for(auto&r:rows)for(ld q:r)y.push_back(q);vector<ld>x(y.size());iota(x.begin(),x.end(),1.0L);Analysis a=analyzeSequence(x,y,horizon,tol);cout<<webOk(string("\"route\":\"sequence\",\"analysis\":")+webAnalysisJson(a,&x,&y)+",\"stats\":"+webStatsJson(computeStats(y)));return 0;}
                if(mode==2){vector<pair<ld,ld>>pairs;for(auto&r:rows)if(r.size()==2)pairs.push_back({r[0],r[1]});sort(pairs.begin(),pairs.end());vector<ld>x,y;for(auto&p:pairs){x.push_back(p.first);y.push_back(p.second);}if(x.size()<3){ cout<<webFail("Could not infer enough x-y pairs."); return 0; }for(size_t i=1;i<x.size();++i)if(fabsl(x[i]-x[i-1])<1e-15L){ cout<<webFail("Duplicate x values in inferred curve."); return 0; }Analysis a=analyzeSequence(x,y,horizon,tol);cout<<webOk(string("\"route\":\"curve\",\"analysis\":")+webAnalysisJson(a,&x,&y));return 0;}
                if(mode==1){
                    const int vars=(int)rows.front().size();
                    if(vars<2||vars>64){cout<<webFail("Inferred variable count is outside the supported range.");return 0;}
                    bool good=true;for(const auto&r:rows)if((int)r.size()!=vars)good=false;
                    if(!good||rows.size()<8){cout<<webFail("Inferred multivariate data has inconsistent row widths or too few rows.");return 0;}
                    MultiAnalysis ma=analyzeMultivariate(rows,horizon,tol);
                    ostringstream body;body<<"\"route\":\"multivariate\",\"analysis\":{";
                    body<<"\"validStructure\":"<<webJsonBool(ma.validStructure)<<",\"fullForecastAvailable\":"<<webJsonBool(ma.fullForecastAvailable)<<",\"variables\":"<<ma.variables<<",\"rows\":"<<ma.rows<<",\"tolerancePct\":"<<webJsonNumber(ma.tolerancePct)<<",\"failureReason\":"<<webJsonEscape(ma.failureReason)<<",\"driver\":"<<webAnalysisJson(ma.driver)<<",\"predictions\":[";
                    for(size_t i=0;i<ma.predictions.size();++i){if(i)body<<',';body<<webJsonArray(ma.predictions[i]);}body<<"]},\"rows\":[";
                    for(size_t i=0;i<rows.size();++i){if(i)body<<',';body<<webJsonArray(rows[i]);}body<<']';
                    cout<<webOk(body.str());return 0;
                }
                cout<<webFail("Could not infer the dataset structure automatically; use an explicit web command."); return 0;
            }
            if(cmd=="--web-curve"){
                int horizon=(argc>=3)?max(1,atoi(argv[2])):4;ld tol=(argc>=4)?stold(argv[3]):-1.0L;ld step=(argc>=5)?stold(argv[4]):1.0L;
                cout<<webRunCurve(horizon,tol,step,false);return 0;
            }
            if(cmd=="--web-families"){
                cout<<webFamilySweepCommand();return 0;
            }
            if(cmd=="--web-family" && argc>=3){
                int degree=(argc>=4)?atoi(argv[3]):1;ld p=(argc>=5)?stold(argv[4]):0;ld p2=(argc>=6)?stold(argv[5]):0;int horizon=(argc>=7)?max(1,atoi(argv[6])):4;ld step=(argc>=8)?stold(argv[7]):1.0L;
                cout<<webFamilyFitCommand(argv[2],degree,p,p2,horizon,step);return 0;
            }
            if(cmd=="--web-multivariate" && argc>=3){
                int vars=max(2,atoi(argv[2]));int horizon=(argc>=4)?max(1,atoi(argv[3])):3;ld tol=(argc>=5)?stold(argv[4]):-1.0L;
                const string text=webReadAll();
                
                istringstream in(text);string line;vector<vector<ld>>rows;bool bad=false;while(getline(in,line)){if(line.find_first_not_of(" \t\r\n")==string::npos)continue;auto v=parseNumberLine(line);if(v.size()!=(size_t)vars){bad=true;break;}rows.push_back(v);}if(bad||rows.size()<8){cout<<webFail("Need valid rows with exactly "+to_string(vars)+" values each; at least 8 rows required.");return 0;}
                MultiAnalysis ma=analyzeMultivariate(rows,horizon,tol);ostringstream body;body<<"\"analysis\":{"<<"\"validStructure\":"<<webJsonBool(ma.validStructure)<<",\"fullForecastAvailable\":"<<webJsonBool(ma.fullForecastAvailable)<<",\"variables\":"<<ma.variables<<",\"rows\":"<<ma.rows<<",\"tolerancePct\":"<<webJsonNumber(ma.tolerancePct)<<",\"failureReason\":"<<webJsonEscape(ma.failureReason)<<",\"driver\":"<<webAnalysisJson(ma.driver)<<",\"predictions\":[";for(size_t i=0;i<ma.predictions.size();++i){if(i)body<<',';body<<webJsonArray(ma.predictions[i]);}body<<"],\"links\":[";for(size_t i=0;i<ma.links.size();++i){if(i)body<<',';auto&r=ma.links[i];body<<'{';body<<"\"found\":"<<webJsonBool(r.found)<<",\"category\":"<<webJsonEscape(r.category)<<",\"logic\":"<<webJsonEscape(r.logic)<<",\"formula\":"<<webJsonEscape(r.formula)<<",\"evidenceScore\":"<<webJsonNumber(r.evidenceScore)<<",\"tolerancePct\":"<<webJsonNumber(r.tolerancePct)<<",\"warning\":"<<webJsonEscape(r.warning)<<",\"model\":"<<webModelJson(r.model,true)<<",\"futurePredictions\":"<<webJsonArray(r.futurePredictions)<<'}';}body<<"]},\"rows\":[";for(size_t i=0;i<rows.size();++i){if(i)body<<',';body<<webJsonArray(rows[i]);}body<<']';cout<<webOk(body.str());return 0;
            }
            if(cmd=="--web-stats"){
                cout<<webStatsFromNumbers(webReadNumbers(webReadAll()));return 0;
            }
            if(cmd=="--web-coordinates" && argc>=3){
                int mode=atoi(argv[2]);int response=(argc>=4)?atoi(argv[3]):1;int angle=(argc>=5)?atoi(argv[4]):0;int horizon=(argc>=6)?max(1,atoi(argv[5])):3;ld tol=(argc>=7)?stold(argv[6]):-1.0L;cout<<webCoordinatesCommand(mode,response,angle,horizon,tol);return 0;
            }
            if(cmd=="--web-roots"){cout<<webRootsCommand();return 0;}
            if(cmd=="--web-file-path" && argc>=3){
                int mode=(argc>=4)?atoi(argv[3]):0;int vars=(argc>=5)?atoi(argv[4]):3;int angle=(argc>=6)?atoi(argv[5]):0;int horizon=(argc>=7)?max(1,atoi(argv[6])):4;ld tol=(argc>=8)?stold(argv[7]):-1.0L;cout<<webFilePathCommand(argv[2],mode,vars,angle,horizon,tol);return 0;
            }
            if(cmd=="--web-file" && argc>=3){
                int mode=atoi(argv[2]);int vars=(argc>=4)?atoi(argv[3]):3;int angle=(argc>=5)?atoi(argv[4]):0;int horizon=(argc>=6)?max(1,atoi(argv[5])):3;ld tol=(argc>=7)?stold(argv[6]):-1.0L;
                if(mode==1){
                    const string text=webReadAll();istringstream in(text);string line;vector<vector<ld>>rows;while(getline(in,line)){if(line.find_first_not_of(" \t\r\n")==string::npos)continue;auto v=parseNumberLine(line);if(v.size()!=(size_t)vars){cout<<webFail("Malformed multivariate file row.");return 0;}rows.push_back(v);}if(rows.size()<8){cout<<webFail("Need at least 8 multivariate rows.");return 0;}MultiAnalysis ma=analyzeMultivariate(rows,horizon,tol);ostringstream body;body<<"\"mode\":1,\"analysis\":{\"validStructure\":"<<webJsonBool(ma.validStructure)<<",\"fullForecastAvailable\":"<<webJsonBool(ma.fullForecastAvailable)<<",\"variables\":"<<ma.variables<<",\"rows\":"<<ma.rows<<",\"tolerancePct\":"<<webJsonNumber(ma.tolerancePct)<<",\"failureReason\":"<<webJsonEscape(ma.failureReason)<<",\"driver\":"<<webAnalysisJson(ma.driver)<<",\"predictions\":[";for(size_t i=0;i<ma.predictions.size();++i){if(i)body<<',';body<<webJsonArray(ma.predictions[i]);}body<<"]},\"rows\":[";for(size_t i=0;i<rows.size();++i){if(i)body<<',';body<<webJsonArray(rows[i]);}body<<']';cout<<webOk(body.str());return 0;
                }
                cout<<webFileCommand(mode,vars,angle,horizon,tol);return 0;
            }
            if(cmd=="--web-cipher" && argc>=3){cout<<webCipherCommand(argv[2]);return 0;}
            if(cmd=="--web-prng" && argc>=3){vector<string>a;for(int i=3;i<argc;++i)a.push_back(argv[i]);cout<<webRandomCommand(argv[2],a);return 0;}
            if(cmd=="--web-selftest"){cout<<webSelfTestCommand();return 0;}
            if(cmd=="--web-puzzle-cube" && argc>=3){int n=atoi(argv[2]);int depth=(argc>=4)?atoi(argv[3]):8;int lim=(argc>=5)?atoi(argv[4]):3000000;cout<<webPuzzleCubeCommand(n,depth,lim);return 0;}
            if(cmd=="--web-puzzle-generic" && argc>=3){int n=atoi(argv[2]);int depth=(argc>=4)?atoi(argv[3]):10;int lim=(argc>=5)?atoi(argv[4]):3000000;cout<<webPuzzleGenericCommand(n,depth,lim);return 0;}
        }

    
        while(true){
            printMainMenu();
            const int method = askIntRange("Select a method (0-10): ", 0, 10, -1);
            if(g_inputClosed) break;
            if(method==0){ cout<<"Goodbye.\n"; break; }
            if(method==1) runSingleSequence();
            else if(method==2) runMultivariate();
            else if(method==3) runCoordinates();
            else if(method==4) runFileMode();
            else if(method==5){printMethodCard(5);cipherMenu();}
            else if(method==6) runStatsMode();
            else if(method==7) runRandomMode();
            else if(method==8) runGraphCurveMode();
            else if(method==9) runSelfTestSuite();
            else if(method==10) runPuzzleMode();
            cout<<"\n--------------------------------------------------------\n";
            if(g_inputClosed || !askYesNo("Return to the main menu?",true)) break;
        }
        return 0;
    }catch(const exception&e){
        cout<<webFail(string("Engine exception: ")+e.what());
        return 2;
    }catch(...){
        cout<<webFail("Unknown engine exception.");
        return 2;
    }
}

