#include <gtest/gtest.h>

#include <numeric>

#include "sfm/viewer/layout.hpp"

using namespace sfm::viewer;  // NOLINT(google-build-using-namespace)

TEST(Layout, DesignSizeMatchesTheFigmaFrame) {
  const Layout l = compute_layout(1440, 900);
  EXPECT_FLOAT_EQ(l.header.h, 52);
  EXPECT_FLOAT_EQ(l.left.w, 248);
  EXPECT_FLOAT_EQ(l.right.w, 320);
  EXPECT_FLOAT_EQ(l.timeline.h, 104);
  EXPECT_FLOAT_EQ(l.viewport.x, 248);
  EXPECT_FLOAT_EQ(l.viewport.y, 52);
  EXPECT_FLOAT_EQ(l.viewport.w, 872);
  EXPECT_FLOAT_EQ(l.viewport.h, 744);
  EXPECT_FLOAT_EQ(l.right.x, 1120);
  EXPECT_FLOAT_EQ(l.timeline.y, 796);
  // Panels tile the window: widths and heights add up.
  EXPECT_FLOAT_EQ(l.left.w + l.viewport.w + l.right.w, 1440);
  EXPECT_FLOAT_EQ(l.header.h + l.viewport.h + l.timeline.h, 900);
  EXPECT_TRUE(l.viewport.contains(700, 400));
  EXPECT_FALSE(l.viewport.contains(100, 400));
}

TEST(Layout, SmallWindowsShrinkTheSidePanelsAndNeverGoNegative) {
  const Layout l = compute_layout(600, 300);
  EXPECT_FLOAT_EQ(l.viewport.w, 160);  // the minimum view
  EXPECT_NEAR(l.left.w / l.right.w, 248.0 / 320.0, 1e-5);
  EXPECT_FLOAT_EQ(l.left.w + l.viewport.w + l.right.w, 600);
  EXPECT_FLOAT_EQ(l.viewport.h, 300 - 52 - 104);
  const Layout tiny = compute_layout(50, 40);
  for (const Rect& r : {tiny.header, tiny.left, tiny.viewport, tiny.right, tiny.timeline}) {
    EXPECT_GE(r.w, 0);
    EXPECT_GE(r.h, 0);
  }
}

TEST(Timeline, MarkPositionsSpanTheScrubber) {
  const auto x = mark_positions(44, 1000, 9);
  ASSERT_EQ(x.size(), 44U);
  EXPECT_FLOAT_EQ(x.front(), 0);
  EXPECT_FLOAT_EQ(x.back(), 991);
  for (std::size_t i = 1; i < x.size(); ++i) EXPECT_GT(x[i], x[i - 1]);
  EXPECT_EQ(mark_positions(1, 1000, 9), std::vector<float>{0});
  EXPECT_TRUE(mark_positions(0, 1000, 9).empty());
  // Clicking a mark selects it.
  for (int i : {0, 7, 43}) {
    EXPECT_EQ(nearest_mark(x[static_cast<std::size_t>(i)] + 4.5F, 44, 1000, 9), i);
  }
  EXPECT_EQ(nearest_mark(-50, 44, 1000, 9), 0);
  EXPECT_EQ(nearest_mark(5000, 44, 1000, 9), 43);
  EXPECT_EQ(nearest_mark(10, 0, 1000, 9), -1);
}

TEST(Timeline, StageFractionsSumToOneAndKeepTinyStagesVisible) {
  // bunny_sift_tuned: load, features, matching, tracks, reconstruction (- BA), BA, export.
  const std::vector<double> s = {0.035, 0.588, 0.056, 0.002, 0.049, 0.271, 0.015};
  const auto f = stage_fractions(s, 0.004F);
  ASSERT_EQ(f.size(), s.size());
  EXPECT_NEAR(std::accumulate(f.begin(), f.end(), 0.0F), 1.0F, 1e-6);
  EXPECT_GE(f[3], 0.0039F);                // tracks (0.2 %) raised to the minimum
  EXPECT_NEAR(f[1], 0.588 / 1.016, 0.01);  // features stays ~58 %
  EXPECT_GT(f[1], f[5]);                   // order kept
  const auto zero = stage_fractions({0, 0, 0, 0}, 0.004F);
  for (float v : zero) EXPECT_FLOAT_EQ(v, 0.25F);
  const auto skipped = stage_fractions({0, 1.0, 0}, 0.004F);  // stages not run stay at 0
  EXPECT_FLOAT_EQ(skipped[0], 0);
  EXPECT_FLOAT_EQ(skipped[1], 1);
}
