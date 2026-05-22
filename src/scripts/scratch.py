from gnss_tid.pointdata import PointData


def main():
    time_limits = ["20150325_234000", "20150325_235200"]
    latitude_limits = [24, 48]
    longitude_limits = [-120, -75]
    return PointData(
        "/disk1/tid/sharon/poly/2015/0325/*.nc",
        latitude_limits=latitude_limits,
        longitude_limits=longitude_limits,
        time_limits=time_limits,
        el_min=30,
        n_jobs=1,
    )


if __name__ == "__main__":
    main()
