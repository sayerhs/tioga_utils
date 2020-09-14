#ifndef NGPLOOPUTILS_H
#define NGPLOOPUTILS_H

#include "ngp_utils/NgpTypes.h"

namespace tioga_nalu {
namespace ngp {

/** Execute the given functor for all entities in a Kokkos parallel loop
 *
 *  The functor is called with one argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity.
 *
 *. @param algName User-defined name for the parallel for loop
 *  @param mesh A STK NGP mesh instance
 *  @param rank Rank for the loop (node, elem, face, etc.)
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 */
template <typename Mesh, typename AlgFunctor>
void run_entity_algorithm(
    const std::string& algName,
    const Mesh& mesh,
    const stk::topology::rank_t rank,
    const stk::mesh::Selector& sel,
    const AlgFunctor algorithm)
{
    using Traits = NGPMeshTraits<Mesh>;
    using TeamPolicy = typename Traits::TeamPolicy;
    using TeamHandleType = typename Traits::TeamHandleType;
    using MeshIndex = typename Traits::MeshIndex;

    const auto& buckets = mesh.get_bucket_ids(rank, sel);
    auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

    Kokkos::parallel_for(
        algName, team_exec, KOKKOS_LAMBDA(const TeamHandleType& team) {
            auto bktId = buckets.device_get(team.league_rank());
            auto& bkt = mesh.get_bucket(rank, bktId);

            const size_t bktLen = bkt.size();
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, bktLen),
                [&](const size_t& bktIndex) {
                    MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
                    algorithm(meshIdx);
                });
        });
}

/** Execute the given functor for all entities and perform global reduction
 *
 *  The functor is called with two argument MeshIndex, a struct containing a
 *  pointer to the NGP bucket and the index into the bucket array for this
 *  entity, and accumulator for reduction.
 *
 *. @param algName User-defined name for the parallel_reduce loop
 *  @param mesh A STK NGP mesh instance
 *  @param rank Rank for the loop (node, elem, face, etc.)
 *  @param sel  STK mesh selector to choose buckets for looping
 *  @param algorithm A functor that will be executed for each entity
 *  @param reduceVal A scalar value that has the reduced value
 */
template<typename Mesh, typename AlgFunctor, typename ReducerType>
void run_entity_par_reduce(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal,
  typename std::enable_if<std::is_arithmetic<ReducerType>::value, int>::type* = nullptr)
{
  using Traits         = NGPMeshTraits<Mesh>;
  using TeamPolicy     = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex      = typename Traits::MeshIndex;

  const auto& buckets = mesh.get_bucket_ids(rank, sel);
  auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, ReducerType& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = mesh.get_bucket(rank, bktId);

      ReducerType bktVal = 0.0;
      const size_t bktLen = bkt.size();
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bktLen),
        [&](const size_t& bktIndex, ReducerType& threadVal) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          algorithm(meshIdx, threadVal);
        }, bktVal);

      Kokkos::single(
        Kokkos::PerTeam(team),
        [&]() {
          teamVal += bktVal;
        });
    }, reduceVal);
}

template<typename Mesh, typename AlgFunctor, typename ReducerType>
void run_entity_par_reduce(
  const std::string& algName,
  const Mesh& mesh,
  const stk::topology::rank_t rank,
  const stk::mesh::Selector& sel,
  const AlgFunctor algorithm,
  ReducerType& reduceVal,
  typename std::enable_if<!std::is_arithmetic<ReducerType>::value, int>::type* = nullptr)
{
  using Traits         = NGPMeshTraits<Mesh>;
  using TeamPolicy     = typename Traits::TeamPolicy;
  using TeamHandleType = typename Traits::TeamHandleType;
  using MeshIndex      = typename Traits::MeshIndex;
  using value_type     = typename ReducerType::value_type;

  const auto& buckets = mesh.get_bucket_ids(rank, sel);
  auto team_exec = TeamPolicy(buckets.size(), Kokkos::AUTO);

  Kokkos::parallel_reduce(
    algName, team_exec,
    KOKKOS_LAMBDA(const TeamHandleType& team, value_type& teamVal) {
      auto bktId = buckets.device_get(team.league_rank());
      auto& bkt = mesh.get_bucket(rank, bktId);

      value_type bktVal;
      const size_t bktLen = bkt.size();
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, bktLen),
        [&](const size_t& bktIndex, value_type& threadVal) {
          MeshIndex meshIdx{&bkt, static_cast<unsigned>(bktIndex)};
          algorithm(meshIdx, threadVal);
        }, ReducerType(bktVal));

      Kokkos::single(
        Kokkos::PerTeam(team),
        [&]() {
          reduceVal.join(teamVal, bktVal);
        });
    }, reduceVal);
}

} // namespace ngp
} // namespace tioga_nalu

#endif /* NGPLOOPUTILS_H */
