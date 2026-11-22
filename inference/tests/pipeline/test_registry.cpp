#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "sfm/config/config.hpp"
#include "sfm/core/error.hpp"
#include "sfm/pipeline/pipeline.hpp"
#include "sfm/pipeline/registry.hpp"

using namespace sfm::pipeline;  // NOLINT(google-build-using-namespace)

TEST(Registry, DefaultMethodsAndUnknownNames) {
  const Registry& r = default_registry();
  EXPECT_TRUE(static_cast<bool>(r.features("orb_cpu")));
  EXPECT_TRUE(static_cast<bool>(r.features("orb_cuda")));
  EXPECT_TRUE(static_cast<bool>(r.features("sift_cpu")));
  EXPECT_TRUE(static_cast<bool>(r.features("sift_cuda")));
  EXPECT_TRUE(static_cast<bool>(r.matching("bf_cpu")));
  EXPECT_TRUE(static_cast<bool>(r.matching("bf_cuda")));
  EXPECT_TRUE(static_cast<bool>(r.ba("lm_schur_cpu")));
  EXPECT_TRUE(static_cast<bool>(r.ba("lm_schur_cuda")));
  try {
    r.features("orb_gpu");
    FAIL() << "no ConfigError";
  } catch (const sfm::ConfigError& e) {
    EXPECT_EQ(std::string(e.what()),
              "features.method: unknown method 'orb_gpu' (known: orb_cpu, orb_cuda, sift_cpu, "
              "sift_cuda)");
  }
  EXPECT_THROW(r.matching("x"), sfm::ConfigError);
  EXPECT_THROW(r.ba("x"), sfm::ConfigError);

  Registry mine;
  mine.add_features("a",
                    [](const sfm::data::Image8&, const sfm::data::Image8*,
                       const sfm::config::FeaturesConfig&) { return sfm::features::Features{}; });
  EXPECT_EQ(mine.features("a")(sfm::data::Image8{}, nullptr, {}).size(), 0u);
}

TEST(Stages, ParseAndPrint) {
  for (const char* s : {"features", "matching", "tracks", "reconstruction", "export"}) {
    EXPECT_EQ(to_string(parse_stage(s)), s);
  }
  EXPECT_LT(parse_stage("features"), parse_stage("export"));
  EXPECT_THROW(parse_stage("ba"), sfm::ConfigError);
}

TEST(Registry, RepositoryConfigsNameKnownMethods) {
  const std::filesystem::path src(SFM_SOURCE_DIR);
  for (const char* name : {"bunny.json", "bunny_gpu.json", "bunny_sift.json", "bunny_sift_tuned.json",
                           "bunny_sift_tuned2.json"}) {
    SCOPED_TRACE(name);
    std::ifstream in(src / "configs" / name);
    std::stringstream text;
    text << in.rdbuf();
    const auto parsed = sfm::config::parse_config(text.str());
    for (const bool gpu : {false, true}) {
      const auto c = resolve_auto_methods(parsed, gpu);
      EXPECT_NO_THROW(default_registry().features(c.features.method));
      EXPECT_NO_THROW(default_registry().matching(c.matching.method));
      EXPECT_NO_THROW(default_registry().ba(c.ba.method));
    }
  }
  const auto gpu = sfm::config::load_config(src / "configs" / "bunny_gpu.json", src);
  EXPECT_EQ(gpu.run.name, "bunny-gpu");  // its own output directory
  EXPECT_EQ(gpu.features.method, "orb_cuda");
  EXPECT_EQ(gpu.matching.method, "bf_cuda");
  EXPECT_EQ(gpu.ba.method, "lm_schur_cuda");
}

TEST(Registry, AutoMethodsResolveByBackend) {
  sfm::config::Config c;
  c.features.method = "auto";
  c.matching.method = "auto";
  c.ba.method = "auto";
  const auto g = resolve_auto_methods(c, true);
  EXPECT_EQ(g.features.method, "sift_cuda");
  EXPECT_EQ(g.matching.method, "bf_cuda");
  EXPECT_EQ(g.ba.method, "lm_schur_cuda");
  const auto n = resolve_auto_methods(c, false);
  EXPECT_EQ(n.features.method, "sift_cpu");
  EXPECT_EQ(n.matching.method, "bf_cpu");
  EXPECT_EQ(n.ba.method, "lm_schur_cpu");
  c.matching.method = "bf_cpu";  // explicit names are kept
  EXPECT_EQ(resolve_auto_methods(c, true).matching.method, "bf_cpu");
}
