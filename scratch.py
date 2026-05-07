from gnss_tid.pointdata import PointData
from gnss_tid.old_pointdata import PointData as old_PointData

def main():
    time_limits = ["20150325_234000", "20150325_235200"]
    latitude_limits = [24, 48]
    longitude_limits = [-120, -75]
    TI = 5
    HEIGHT = 200
    WINDOW = 4
    STEP = 2

    points = PointData(
        "/disk1/tid/sharon/poly/2015/0325/*.nc",
        latitude_limits=latitude_limits,
        longitude_limits=longitude_limits,
        time_limits=time_limits,
        el_min=30,
        n_jobs=1
    )

    # points = old_PointData(
    #     ["/disk1/tid/data/2015_0325T0000-0326T0000_all0325.yaml_30el_30s_ra.h5"],
    #     latitude_limits=latitude_limits,
    #     longitude_limits=longitude_limits,
    #     time_limits=time_limits,
    # )



if __name__ == "__main__":
    main()
