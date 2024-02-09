import torch


def project(source, target, query):
    diff_vec = source - target
    projection = (diff_vec @ query.T) / torch.linalg.vector_norm(diff_vec)
    return projection

def reconfigure_dist(start_dist, end_dist):
    dist_start1, dist_end1 = start_dist[0][0].item(), end_dist[0][0].item()
    dist_start2, dist_end2 = start_dist[1][1].item(), end_dist[1][1].item()

    return dist_start1, dist_end1, dist_start2, dist_end2

