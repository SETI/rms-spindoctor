================
Image Navigation
================

Navigation is the first stage of the pipeline: it compares an image against a
model of what the spacecraft should have been looking at and reports a corrected
pointing for the frame. The chapters below cover running the navigation program,
choosing the models and techniques it uses, the files it writes, the programs
that summarize its results, and the tasks the navigation cloud worker reads.

.. toctree::
   :maxdepth: 2

   user_guide_navigation_running
   user_guide_navigation_models
   user_guide_navigation_outputs
   user_guide_navigation_troubleshooting
   user_guide_metadata
   user_guide_results_index
   user_guide_statistics
   user_guide_consolidate_metadata
   user_guide_simulated_images
   user_guide_navigation_cloud_tasks
