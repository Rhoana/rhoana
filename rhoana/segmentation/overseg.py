import mahotas

def overseg_by_watershed(probabilities,
                         minimum_suppression_window=50,
                         distance_transform=True):
    '''Oversegment a probability map into regions with watershed.

    Input is a membrane probability map (high = membrane), output is a
    labeled slice or volume of the same dimension as the input.

    minimum_suppression_window - suppresion window (circular) for local
    minima to use as watershed seeds.
    '''

    ws, count = mahotas.watershed(probabilities, minimum_suppression_window)
    if distance_transform:
        dists = mahotas.distance_transform(ws)
        ws, count = mahotas.watershed(dists)

    return ws
