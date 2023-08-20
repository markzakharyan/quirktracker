import quirk_multiprocessing, background_multiprocessing

passed = quirk_multiprocessing.process()
print(f"passed: {passed}")

background_multiprocessing.process(passed)